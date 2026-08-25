# Two stages: build the release, then run it on a runtime with no build tools.
# The release is the artifact both this image and scripts/run-bwrap.sh execute,
# so the sandboxed run and the container run the same bytes.

ARG ELIXIR_IMAGE=elixir:1.18-otp-27-slim
ARG RUNTIME_IMAGE=debian:bookworm-slim

FROM ${ELIXIR_IMAGE} AS build

ENV MIX_ENV=prod

# The slim image ships no CA bundle, and hex fetches over TLS. Without this,
# `mix local.hex` fails inside :pubkey_os_cacerts with :no_cacerts_found.
RUN apt-get update  && apt-get install -y --no-install-recommends ca-certificates git  && rm -rf /var/lib/apt/lists/*

RUN mix local.hex --force && mix local.rebar --force

WORKDIR /src
COPY mix.exs mix.lock ./
RUN mix deps.get --only prod

# config comes before deps.compile, not after. Livebook's own mix.exs reads
# `Application.get_all_env(:livebook)` in `application/0`, so compiling the dep
# without this project's config raises on a nil endpoint entry.
COPY config config
RUN mix deps.compile

COPY lib lib
COPY rel rel
RUN mix release --overwrite

FROM ${RUNTIME_IMAGE} AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libncurses6 libstdc++6 openssl \
 && rm -rf /var/lib/apt/lists/*

# Livebook writes notebooks and its own data under this path, and it is the one
# writable directory the sandbox profile binds read-write.
ENV LIVEBOOK_DATA_PATH=/data \
    LIVEBOOK_IP=0.0.0.0 \
    LIVEBOOK_PORT=8080 \
    LANG=C.UTF-8

RUN useradd --create-home --uid 10001 livebook \
 && mkdir -p /data && chown livebook:livebook /data

COPY --from=build --chown=livebook:livebook /src/_build/prod/rel/service_livebook /app

USER livebook
WORKDIR /app
EXPOSE 8080

# `start` runs in the foreground, which is what a container init expects.
ENTRYPOINT ["/app/bin/service_livebook"]
CMD ["start"]
