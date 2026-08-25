defmodule ServiceLivebook.MixProject do
  use Mix.Project

  def project do
    [
      app: :service_livebook,
      version: "0.1.0",
      elixir: "~> 1.18",
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [{:livebook, "~> 0.19.9"}]
  end
end
