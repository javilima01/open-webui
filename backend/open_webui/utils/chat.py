import time
import logging
import sys

from aiocache import cached
from typing import Any, Optional
import random
import json
import inspect
import uuid
import asyncio

from fastapi import Request, status
from starlette.responses import Response, StreamingResponse, JSONResponse


from open_webui.models.users import UserModel

from open_webui.socket.main import (
    sio,
    get_event_call,
    get_event_emitter,
)
from open_webui.functions import generate_function_chat_completion

from open_webui.routers.openai import (
    generate_chat_completion as generate_openai_chat_completion,
)

from open_webui.routers.ollama import (
    generate_chat_completion as generate_ollama_chat_completion,
)

from open_webui.routers.pipelines import (
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)

from open_webui.models.functions import Functions
from open_webui.models.models import Models


from open_webui.utils.plugin import (
    load_function_module_by_id,
    get_function_module_from_cache,
)
from open_webui.utils.models import get_all_models, check_model_access
from open_webui.utils.payload import convert_payload_openai_to_ollama
from open_webui.utils.response import (
    convert_response_ollama_to_openai,
    convert_streaming_response_ollama_to_openai,
)
from open_webui.utils.filter import (
    get_sorted_filter_ids,
    process_filter_functions,
)
from open_webui.utils.misc import (
    openai_chat_chunk_message_template,
    openai_chat_completion_message_template,
)

from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL, BYPASS_MODEL_ACCESS_CONTROL


logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


# UI event types that should NOT be sent to API clients
UI_EVENT_TYPES = frozenset([
    "status",
    "citation",
    "source",
    "chat:title",
    "chat:tags",
    "chat:message:delta",
    "chat:message:error",
    "chat:message:follow_ups",
    "chat:completion",
    "chat:tasks:cancel",
    "message",
    "replace",
    "embeds",
    "files",
])


def is_api_request(metadata: dict) -> bool:
    """
    Determine if a request is an API request (no UI session) vs a UI request.
    
    API requests lack session_id, chat_id, and message_id metadata that UI
    requests provide via WebSocket connections. These fields may exist in metadata
    but be None or empty for API requests.
    """
    if not metadata:
        return True
    
    session_id = metadata.get("session_id")
    chat_id = metadata.get("chat_id")
    message_id = metadata.get("message_id")
    
    # All three must be present and truthy for a UI request
    return not (session_id and chat_id and message_id)


def is_openai_compatible_chunk(data: dict) -> bool:
    """
    Check if a data chunk is OpenAI-compatible (has choices with delta).
    """
    if not isinstance(data, dict):
        return False
    choices = data.get("choices", [])
    if not choices:
        return False
    # Valid OpenAI chunk has delta in choices
    return "delta" in choices[0] or "message" in choices[0]


def is_ui_event(data: dict) -> bool:
    """
    Check if data is a UI-specific event that should be filtered for API clients.
    Handles both direct format {"type": "status"} and wrapped format {"event": {"type": "status"}}
    """
    if not isinstance(data, dict):
        return False
    
    # Check for wrapped event format: {"event": {"type": "..."}}
    if "event" in data and isinstance(data["event"], dict):
        event_type = data["event"].get("type", "")
        return event_type in UI_EVENT_TYPES
    
    # Check for direct format: {"type": "..."}
    event_type = data.get("type", "")
    return event_type in UI_EVENT_TYPES


def extract_content_from_ui_event(data: dict) -> Optional[str]:
    """
    Extract text content from UI event if it contains actual token content.
    Returns None if no content should be emitted.
    Handles both direct and wrapped event formats.
    """
    # Handle wrapped event format: {"event": {"type": "message", "data": {"content": "..."}}}
    if "event" in data and isinstance(data["event"], dict):
        event = data["event"]
        event_type = event.get("type", "")
        event_data = event.get("data", {})
        
        if event_type == "message" and isinstance(event_data, dict):
            return event_data.get("content", "")
        return None
    
    # Handle direct format: {"type": "message", "data": {"content": "..."}}
    event_type = data.get("type", "")
    event_data = data.get("data", {})
    
    if event_type == "message":
        return event_data.get("content", "")
    
    return None


async def wrap_streaming_response_for_api(
    response: StreamingResponse,
    model_id: str,
) -> StreamingResponse:
    """
    Wrap a streaming response to filter out UI-specific events for API clients.
    Only emits OpenAI-compatible streaming chunks.
    """
    
    async def filtered_stream():
        log.info(f"[API_FILTER] Starting filtered_stream for model {model_id}")
        chunk_count = 0
        async for chunk in response.body_iterator:
            chunk_count += 1
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            
            log.debug(f"[API_FILTER] Raw chunk #{chunk_count}: {chunk[:200] if len(chunk) > 200 else chunk}")
            
            # Handle SSE format - may have multiple lines in one chunk
            for line in chunk.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                # Handle [DONE] marker
                if line == "data: [DONE]":
                    log.info(f"[API_FILTER] Received [DONE] marker")
                    # Emit final chunk with finish_reason
                    finish_chunk = openai_chat_chunk_message_template(model_id, "")
                    finish_chunk["choices"][0]["finish_reason"] = "stop"
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    continue
                
                # Parse data lines
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        log.debug(f"[API_FILTER] Skipping non-JSON line: {line[:100]}")
                        continue
                    
                    # Check if this is already an OpenAI-compatible chunk
                    if is_openai_compatible_chunk(data):
                        log.debug(f"[API_FILTER] Passing through OpenAI chunk")
                        yield f"data: {json.dumps(data)}\n\n"
                        continue
                    
                    # Check if it's a UI event (handles both wrapped and direct format)
                    if is_ui_event(data):
                        # Try to extract content from UI events
                        content = extract_content_from_ui_event(data)
                        if content:
                            log.info(f"[API_FILTER] Extracted content from UI event: {content[:50]}...")
                            # Emit the content as an OpenAI-compatible chunk
                            chunk_data = openai_chat_chunk_message_template(model_id, content)
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                        else:
                            log.debug(f"[API_FILTER] Filtering out UI event: {data.get('event', {}).get('type') or data.get('type')}")
                        # Skip UI events without content (status updates, etc.)
                        continue
                    
                    # For any other dict data, try to emit as-is if it looks like OpenAI format
                    if "choices" in data:
                        log.debug(f"[API_FILTER] Passing through data with choices")
                        yield f"data: {json.dumps(data)}\n\n"
        
        log.info(f"[API_FILTER] Finished filtered_stream, processed {chunk_count} chunks")
    
    return StreamingResponse(
        filtered_stream(),
        media_type="text/event-stream",
        headers=dict(response.headers) if hasattr(response, 'headers') else {},
        background=response.background if hasattr(response, 'background') else None,
    )


def wrap_response_for_api(response: Any, model_id: str) -> dict:
    """
    Ensure a non-streaming response is in strict OpenAI format for API clients.
    """
    if isinstance(response, dict):
        # Already a dict, check if it's OpenAI-compatible
        if "choices" in response and response.get("object") in ["chat.completion", "chat.completion.chunk"]:
            # Check if the message content contains UI events (pipeline concatenated output)
            choices = response.get("choices", [])
            if choices and "message" in choices[0]:
                msg = choices[0]["message"]
                content = msg.get("content", "")
                # Check if content looks like concatenated UI events
                if isinstance(content, str) and "{'event':" in content:
                    # Parse and extract actual message content from concatenated events
                    extracted_content = extract_content_from_concatenated_events(content)
                    if extracted_content:
                        choices[0]["message"]["content"] = extracted_content
            
            # Filter out any UI-specific fields
            clean_response = {
                "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
                "object": "chat.completion",
                "created": response.get("created", int(time.time())),
                "model": response.get("model", model_id),
                "choices": choices,
            }
            if "usage" in response:
                clean_response["usage"] = response["usage"]
            return clean_response
        
        # Check if it's a UI event with content (handles both wrapped and direct format)
        if is_ui_event(response):
            content = extract_content_from_ui_event(response)
            if content:
                return openai_chat_completion_message_template(model_id, content)
            # Return empty completion for events without content
            return openai_chat_completion_message_template(model_id, "")
        
        # Check for error responses
        if "error" in response:
            return response
        
        # Try to extract message content from various formats
        if "message" in response:
            msg = response["message"]
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)
            return openai_chat_completion_message_template(model_id, content)
        
        if "content" in response:
            return openai_chat_completion_message_template(model_id, response["content"])
        
        # Fallback: return as-is (might be an error or special response)
        return response
    
    # For string responses, wrap in OpenAI format
    if isinstance(response, str):
        return openai_chat_completion_message_template(model_id, response)
    
    # For other types, return as-is
    return response


def extract_content_from_concatenated_events(content: str) -> str:
    """
    Extract actual message content from a string that contains concatenated UI events.
    This handles the case where non-streaming pipeline responses concatenate all events.
    """
    import re
    
    extracted_content = ""
    
    # Try to find event dictionaries in the content string
    # Pattern to match {'event': {...}} or {"event": {...}}
    # We look for message events specifically
    
    # Simple approach: look for content within message events
    # Pattern: 'content': 'actual message' or "content": "actual message"
    
    # Find all potential JSON-like event objects
    try:
        # Split by }{ to separate concatenated dicts (common pattern)
        parts = re.split(r'\}\s*\{', content)
        
        for i, part in enumerate(parts):
            # Reconstruct the JSON object
            if i > 0:
                part = '{' + part
            if i < len(parts) - 1:
                part = part + '}'
            
            try:
                # Try to parse as JSON (handle single quotes by replacing)
                json_str = part.replace("'", '"').replace('False', 'false').replace('True', 'true').replace('None', 'null')
                data = json.loads(json_str)
                
                # Check if this is a message event
                if isinstance(data, dict) and "event" in data:
                    event = data["event"]
                    if isinstance(event, dict) and event.get("type") == "message":
                        event_data = event.get("data", {})
                        if isinstance(event_data, dict):
                            msg_content = event_data.get("content", "")
                            if msg_content:
                                extracted_content += msg_content
            except (json.JSONDecodeError, ValueError):
                continue
    except Exception:
        pass
    
    return extracted_content if extracted_content else content


async def generate_direct_chat_completion(
    request: Request,
    form_data: dict,
    user: Any,
    models: dict,
):
    log.info("generate_direct_chat_completion")

    metadata = form_data.pop("metadata", {})

    user_id = metadata.get("user_id")
    session_id = metadata.get("session_id")
    request_id = str(uuid.uuid4())  # Generate a unique request ID

    event_caller = get_event_call(metadata)

    channel = f"{user_id}:{session_id}:{request_id}"
    logging.info(f"WebSocket channel: {channel}")

    if form_data.get("stream"):
        q = asyncio.Queue()

        async def message_listener(sid, data):
            """
            Handle received socket messages and push them into the queue.
            """
            await q.put(data)

        # Register the listener
        sio.on(channel, message_listener)

        # Start processing chat completion in background
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )

        log.info(f"res: {res}")

        if res.get("status", False):
            # Define a generator to stream responses
            async def event_generator():
                nonlocal q
                try:
                    while True:
                        data = await q.get()  # Wait for new messages
                        if isinstance(data, dict):
                            if "done" in data and data["done"]:
                                break  # Stop streaming when 'done' is received

                            yield f"data: {json.dumps(data)}\n\n"
                        elif isinstance(data, str):
                            if "data:" in data:
                                yield f"{data}\n\n"
                            else:
                                yield f"data: {data}\n\n"
                except Exception as e:
                    log.debug(f"Error in event generator: {e}")
                    pass

            # Define a background task to run the event generator
            async def background():
                try:
                    del sio.handlers["/"][channel]
                except Exception as e:
                    pass

            # Return the streaming response
            return StreamingResponse(
                event_generator(), media_type="text/event-stream", background=background
            )
        else:
            raise Exception(str(res))
    else:
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )

        if "error" in res and res["error"]:
            raise Exception(res["error"])

        return res


async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user: Any,
    bypass_filter: bool = False,
):
    log.debug(f"generate_chat_completion: {form_data}")
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    if hasattr(request.state, "metadata"):
        if "metadata" not in form_data:
            form_data["metadata"] = request.state.metadata
        else:
            form_data["metadata"] = {
                **form_data["metadata"],
                **request.state.metadata,
            }

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
        log.debug(f"direct connection to model: {models}")
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise Exception("Model not found")

    model = models[model_id]

    if getattr(request.state, "direct", False):
        return await generate_direct_chat_completion(
            request, form_data, user=user, models=models
        )
    else:
        # Check if user has access to the model
        if not bypass_filter and user.role == "user":
            try:
                check_model_access(user, model)
            except Exception as e:
                raise e

        if model.get("owned_by") == "arena":
            model_ids = model.get("info", {}).get("meta", {}).get("model_ids")
            filter_mode = model.get("info", {}).get("meta", {}).get("filter_mode")
            if model_ids and filter_mode == "exclude":
                model_ids = [
                    model["id"]
                    for model in list(request.app.state.MODELS.values())
                    if model.get("owned_by") != "arena" and model["id"] not in model_ids
                ]

            selected_model_id = None
            if isinstance(model_ids, list) and model_ids:
                selected_model_id = random.choice(model_ids)
            else:
                model_ids = [
                    model["id"]
                    for model in list(request.app.state.MODELS.values())
                    if model.get("owned_by") != "arena"
                ]
                selected_model_id = random.choice(model_ids)

            form_data["model"] = selected_model_id

            if form_data.get("stream") == True:

                async def stream_wrapper(stream):
                    yield f"data: {json.dumps({'selected_model_id': selected_model_id})}\n\n"
                    async for chunk in stream:
                        yield chunk

                response = await generate_chat_completion(
                    request, form_data, user, bypass_filter=True
                )
                return StreamingResponse(
                    stream_wrapper(response.body_iterator),
                    media_type="text/event-stream",
                    background=response.background,
                )
            else:
                return {
                    **(
                        await generate_chat_completion(
                            request, form_data, user, bypass_filter=True
                        )
                    ),
                    "selected_model_id": selected_model_id,
                }

        if model.get("pipe"):
            # Below does not require bypass_filter because this is the only route the uses this function and it is already bypassing the filter
            response = await generate_function_chat_completion(
                request, form_data, user=user, models=models
            )
            
            # For API requests (no UI session), ensure OpenAI-compatible response format
            metadata = form_data.get("metadata", {})
            api_request = is_api_request(metadata)
            log.info(f"[API_FILTER] Pipeline model detected: {model_id}")
            log.info(f"[API_FILTER] is_api_request: {api_request}")
            log.info(f"[API_FILTER] metadata: session_id={metadata.get('session_id')}, chat_id={metadata.get('chat_id')}, message_id={metadata.get('message_id')}")
            log.info(f"[API_FILTER] response type: {type(response)}")
            
            if api_request:
                log.info(f"[API_FILTER] Wrapping response for API client")
                if isinstance(response, StreamingResponse):
                    return await wrap_streaming_response_for_api(response, model_id)
                else:
                    return wrap_response_for_api(response, model_id)
            else:
                log.info(f"[API_FILTER] NOT wrapping response (UI request)")
            
            return response
        if model.get("owned_by") == "ollama":
            # Using /ollama/api/chat endpoint
            form_data = convert_payload_openai_to_ollama(form_data)
            response = await generate_ollama_chat_completion(
                request=request,
                form_data=form_data,
                user=user,
                bypass_filter=bypass_filter,
            )
            if form_data.get("stream"):
                response.headers["content-type"] = "text/event-stream"
                return StreamingResponse(
                    convert_streaming_response_ollama_to_openai(response),
                    headers=dict(response.headers),
                    background=response.background,
                )
            else:
                return convert_response_ollama_to_openai(response)
        else:
            return await generate_openai_chat_completion(
                request=request,
                form_data=form_data,
                user=user,
                bypass_filter=bypass_filter,
            )


chat_completion = generate_chat_completion


async def chat_completed(request: Request, form_data: dict, user: Any):
    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    data = form_data
    model_id = data["model"]
    if model_id not in models:
        raise Exception("Model not found")

    model = models[model_id]

    try:
        data = await process_pipeline_outlet_filter(request, data, user, models)
    except Exception as e:
        return Exception(f"Error: {e}")

    metadata = {
        "chat_id": data["chat_id"],
        "message_id": data["id"],
        "filter_ids": data.get("filter_ids", []),
        "session_id": data["session_id"],
        "user_id": user.id,
    }

    extra_params = {
        "__event_emitter__": get_event_emitter(metadata),
        "__event_call__": get_event_call(metadata),
        "__user__": user.model_dump() if isinstance(user, UserModel) else {},
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }

    try:
        filter_functions = [
            Functions.get_function_by_id(filter_id)
            for filter_id in get_sorted_filter_ids(
                request, model, metadata.get("filter_ids", [])
            )
        ]

        result, _ = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="outlet",
            form_data=data,
            extra_params=extra_params,
        )
        return result
    except Exception as e:
        return Exception(f"Error: {e}")


async def chat_action(request: Request, action_id: str, form_data: dict, user: Any):
    if "." in action_id:
        action_id, sub_action_id = action_id.split(".")
    else:
        sub_action_id = None

    action = Functions.get_function_by_id(action_id)
    if not action:
        raise Exception(f"Action not found: {action_id}")

    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    data = form_data
    model_id = data["model"]

    if model_id not in models:
        raise Exception("Model not found")
    model = models[model_id]

    __event_emitter__ = get_event_emitter(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )
    __event_call__ = get_event_call(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )

    function_module, _, _ = get_function_module_from_cache(request, action_id)

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        valves = Functions.get_function_valves_by_id(action_id)
        function_module.valves = function_module.Valves(**(valves if valves else {}))

    if hasattr(function_module, "action"):
        try:
            action = function_module.action

            # Get the signature of the function
            sig = inspect.signature(action)
            params = {"body": data}

            # Extra parameters to be passed to the function
            extra_params = {
                "__model__": model,
                "__id__": sub_action_id if sub_action_id is not None else action_id,
                "__event_emitter__": __event_emitter__,
                "__event_call__": __event_call__,
                "__request__": request,
            }

            # Add extra params in contained in function signature
            for key, value in extra_params.items():
                if key in sig.parameters:
                    params[key] = value

            if "__user__" in sig.parameters:
                __user__ = user.model_dump() if isinstance(user, UserModel) else {}

                try:
                    if hasattr(function_module, "UserValves"):
                        __user__["valves"] = function_module.UserValves(
                            **Functions.get_user_valves_by_id_and_user_id(
                                action_id, user.id
                            )
                        )
                except Exception as e:
                    log.exception(f"Failed to get user values: {e}")

                params = {**params, "__user__": __user__}

            if inspect.iscoroutinefunction(action):
                data = await action(**params)
            else:
                data = action(**params)

        except Exception as e:
            return Exception(f"Error: {e}")

    return data
