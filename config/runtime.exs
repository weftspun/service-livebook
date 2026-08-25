import Config

# Upstream's own `config/runtime.exs` is one call, and a runtime config file
# cannot `import_config`. Calling the same function is the closest thing to
# importing it: the LIVEBOOK_* environment variables are read by Livebook, here,
# at boot, and this file decides none of them.
# LIVEBOOK_HOME has to be set before the application starts, not inside `start/0`:
# `:livebook` is a dependency, so Mix and the release both start it for us, and by the
# time any of this project's code runs Livebook has already read its home. A config file
# is the only place early enough.
if System.get_env("LIVEBOOK_HOME") == nil do
  notebooks = Path.expand("notebooks", File.cwd!())
  if File.dir?(notebooks), do: System.put_env("LIVEBOOK_HOME", notebooks)
end

Livebook.config_runtime()
