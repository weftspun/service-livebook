defmodule ServiceLivebook do
  @moduledoc """
  Start Livebook in this project's VM.

  Livebook is a dependency here, not a checkout. `Application.ensure_all_started/1`
  boots the same OTP application the release would, and it reads its own
  configuration, so nothing in this module restates a Livebook setting.
  """

  @doc "Start Livebook and return the URL it is serving."
  def start do
    {:ok, _} = Application.ensure_all_started(:livebook)
    LivebookWeb.Endpoint.access_url()
  end
end
