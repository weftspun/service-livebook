import Config

# Livebook reads its own settings through `Application.get_all_env(:livebook)`, and a
# dependency's config is not loaded by Mix. Importing the config it ships in the hex
# package is what makes it a dependency rather than a rewrite: every value stays
# upstream's, and this file states no setting of its own.
#
# THE GUARD IS NOT DEFENSIVE PROGRAMMING, IT IS THE ONLY WAY A FRESH CLONE WORKS. Mix
# evaluates this file before it fetches anything, so on a checkout with no `deps/` the
# import fails and `mix deps.get` cannot run -- the command that would create the file it
# is looking for. A fresh `repo sync` of this project hit exactly that, with
#
#     ** (File.Error) could not read file ".../deps/livebook/config/config.exs"
#
# and no way forward except deleting this line by hand. The absence is a real state with a
# correct response, which is to configure nothing yet, rather than an error.
livebook_config = Path.expand("../deps/livebook/config/config.exs", __DIR__)

if File.exists?(livebook_config) do
  import_config livebook_config
end
