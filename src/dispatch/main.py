import time
import logging
from os import path
from uuid import uuid1
from typing import Optional, Final
from contextvars import ContextVar

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic.error_wrappers import ValidationError

from sentry_asgi import SentryMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import inspect
from sqlalchemy.orm import scoped_session
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.routing import compile_path

from starlette.responses import Response, StreamingResponse, FileResponse
from starlette.staticfiles import StaticFiles

from .api import api_router
from .common.utils.cli import install_plugins, install_plugin_events
from .config import (
    STATIC_DIR,
)
from .database.core import engine, sessionmaker
from .extensions import configure_extensions
from .logging import configure_logging
from .metrics import provider as metric_provider
from .rate_limiter import limiter


log = logging.getLogger(__name__)

# we configure the logging level and format
configure_logging()

# we configure the extensions such as Sentry
configure_extensions()


async def not_found(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content={"detail": [{"msg": "Not Found."}]}
    )


exception_handlers = {404: not_found}

# we create the ASGI for the app
app = FastAPI(exception_handlers=exception_handlers, openapi_url="")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# we create the ASGI for the frontend
frontend = FastAPI(openapi_url="")


@frontend.middleware("http")
async def default_page(request, call_next):
    response = await call_next(request)
    if response.status_code == 404:
        if STATIC_DIR:
            return FileResponse(path.join(STATIC_DIR, "index.html"))
    return response


# we create the Web API framework
async def db_session_middleware(request: Request, call_next):
    request_id = str(uuid1())

    # we create a per-request id such that we can ensure that our session is scoped for a particular request.
    # see: https://github.com/tiangolo/fastapi/issues/726
    ctx_token = _request_id_ctx_var.set(request_id)
    path_params = get_path_params_from_request(request)

    # if this call is organization specific set the correct search path
    organization_slug = path_params.get("organization", "default")
    request.state.organization = organization_slug
    schema = f"dispatch_organization_{organization_slug}"
    # validate slug exists
    schema_names = inspect(engine).get_schema_names()
    if schema in schema_names:
        # SECURITY FIX: Broken Access Control - Verify user authorization before setting organization schema
        # Vulnerability: The organization path parameter is user-controlled. Previously, only schema existence
        # was checked, allowing any authenticated user to access any organization's data by changing the path.
        # Fix: Extract the authenticated user from the JWT token and verify organization membership before
        # setting the schema translation map. This preserves multi-tenant schema isolation while preventing
        # horizontal privilege escalation.

        # Skip authorization check for default organization (public/shared resources)
        if organization_slug != "default":
            # Extract user identity from JWT token in Authorization header
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                _request_id_ctx_var.reset(ctx_token)
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": [{"msg": "Authentication required to access organization resources"}]},
                )

            token = auth_header[7:]
            try:
                # Verify JWT signature to prevent token forgery and extract user identity
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user_email = payload.get("sub")
            except Exception:
                _request_id_ctx_var.reset(ctx_token)
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": [{"msg": "Invalid or expired authentication token"}]},
                )

            if not user_email:
                _request_id_ctx_var.reset(ctx_token)
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": [{"msg": "Unable to determine user identity from token"}]},
                )

            # Verify user has access to the requested organization using a session on the default schema
            # Uses parameterized query to prevent SQL injection
            # This prevents horizontal privilege escalation by checking membership before schema selection
            verification_session = sessionmaker(bind=engine)()
            try:
                has_access = verification_session.execute(
                    text("SELECT 1 FROM user_organization WHERE user_email = :email AND organization_slug = :slug"),
                    {"email": user_email, "slug": organization_slug}
                ).fetchone()

                if not has_access:
                    _request_id_ctx_var.reset(ctx_token)
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"detail": [{"msg": "Access denied: user is not a member of this organization"}]},
                    )
            finally:
                verification_session.close()

        # add correct schema mapping depending on the request
        schema_engine = engine.execution_options(
            schema_translate_map={
                None: schema,
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": [{"msg": f"Unknown database schema name: {schema}"}]},
        )

    try:
        session = scoped_session(sessionmaker(bind=schema_engine), scopefunc=get_request_id)
        request.state.db = session()
        response = await call_next(request)
    except Exception as e:
        raise e from None
    finally:
        request.state.db.close()

    _request_id_ctx_var.reset(ctx_token)
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000 ; includeSubDomains"
    return response


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path_template = get_path_template(request)

        method = request.method
        tags = {"method": method, "endpoint": path_template}

        try:
            start = time.perf_counter()
            response = await call_next(request)
            elapsed_time = time.perf_counter() - start
            tags.update({"status_code": response.status_code})
            metric_provider.counter("server.call.counter", tags=tags)
            metric_provider.timer("server.call.elapsed", value=elapsed_time, tags=tags)
            log.debug(f"server.call.elapsed.{path_template}: {elapsed_time}")
        except Exception as e:
            metric_provider.counter("server.call.exception.counter", tags=tags)
            raise e from None
        return response


class ExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StreamingResponse:
        try:
            response = await call_next(request)
        except ValidationError as e:
            log.exception(e)
            response = JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": e.errors()}
            )
        except ValueError as e:
            log.exception(e)
            response = JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": [{"msg": "Unknown", "loc": ["Unknown"], "type": "Unknown"}]},
            )
        except Exception as e:
            log.exception(e)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": [{"msg": "Unknown", "loc": ["Unknown"], "type": "Unknown"}]},
            )

        return response


# we add a middleware class for logging exceptions to Sentry
api.add_middleware(SentryMiddleware)

# we add a middleware class for capturing metrics using Dispatch's metrics provider
api.add_middleware(MetricsMiddleware)

api.add_middleware(ExceptionMiddleware)

# we install all the plugins
install_plugins()

# we add all the plugin event API routes to the API router
install_plugin_events(api_router)

# we add all API routes to the Web API framework
api.include_router(api_router)

# we mount the frontend and app
if STATIC_DIR and path.isdir(STATIC_DIR):
    frontend.mount("/", StaticFiles(directory=STATIC_DIR), name="app")

app.mount("/api/v1", app=api)
app.mount("/", app=frontend)
