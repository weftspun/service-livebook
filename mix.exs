defmodule ServiceLivebook.MixProject do
  use Mix.Project

  def project do
    [
      app: :service_livebook,
      version: "0.1.0",
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      aliases: aliases(),
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # `mix serve` points LIVEBOOK_HOME at this repository's notebooks, so the four loops
  # are on the home screen rather than behind a file dialog.
  defp aliases do
    [serve: ["run --no-halt -e 'IO.puts(ServiceLivebook.start())'"]]
  end

  defp deps do
    [
      {:livebook, "~> 0.19.9"},
      {:mox, "~> 1.2", only: :test}
    ]
  end
end
