import Config

# Livebook reads its own settings through `Application.get_all_env(:livebook)`,
# and a dependency's config is not loaded by Mix. Importing the config it ships
# in the hex package is what makes it a dependency rather than a rewrite: every
# value stays upstream's, and this file states no setting of its own.
import_config "../deps/livebook/config/config.exs"
