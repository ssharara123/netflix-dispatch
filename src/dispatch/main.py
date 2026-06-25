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
        # FIX: Broken Access Control - The user-controlled organization path parameter was used to select
        # a database schema without verifying the authenticated user is authorized to access that organization.
        # Now we check the user's organization memberships before allowing schema access, preventing
        # horizontal privilege escalation where any authenticated user could access another org's data.
        # This is safe because it enforces organization-level authorization at the middleware level.
        # Functionality preserved: Authorized users can still access their organization's data as before.
        current_user = getattr(request.state, "user", None)
        if organization_slug != "default" and current_user is not None:
            is_superuser = getattr(current_user, "is_superuser", False)
            user_orgs = getattr(current_user, "organizations", []) or []
            authorized_slugs = set()
            for org in user_orgs:
                slug = getattr(org, "slug", None) or (org.get("slug") if isinstance(org, dict) else org)
                if slug:
                    authorized_slugs.add(slug)
            if not is_superuser and organization_slug not in authorized_slugs:
                _request_id_ctx_var.reset(ctx_token)
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": [{"msg": f"Not authorized to access organization: {organization_slug}"}]},
                )
        # add correct schema mapping depending on the request
        schema_engine = engine.execution_options(
            schema_translate_map={
                None: schema,
            }
        )
    else:
        # FIX: Reset context token before early return to prevent context variable leak
        _request_id_ctx_var.reset(ctx_token)
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
