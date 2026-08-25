import Config

# Upstream's own `config/runtime.exs` is one call, and a runtime config file
# cannot `import_config`. Calling the same function is the closest thing to
# importing it: the LIVEBOOK_* environment variables are read by Livebook, here,
# at boot, and this file decides none of them.
Livebook.config_runtime()
