defmodule ServiceLivebook do
  @moduledoc """
  Start Livebook in this project's VM.

  Livebook is a dependency here, not a checkout. `Application.ensure_all_started/1`
  boots the same OTP application the release would, and it reads its own
  configuration, so nothing in this module restates a Livebook setting.
  """

  @doc """
  Start Livebook and return the URL it is serving.

  `config/runtime.exs` points `LIVEBOOK_HOME` at this repository's `notebooks/` unless the
  caller already set it, so the four loop notebooks are on the home screen. That happens in
  a config file rather than here because `:livebook` is a dependency and is already started
  by the time this function runs.
  """
  def start do
    {:ok, _} = Application.ensure_all_started(:livebook)
    LivebookWeb.Endpoint.access_url()
  end

  # Resolved at compile time against this file, because a release moves the beam files
  # and there is no notebooks directory beside them. The caller sets LIVEBOOK_HOME in
  # that case, and the File.dir? check below is what makes the absence harmless.
  @notebooks Path.expand("../notebooks", __DIR__)

  @doc "The notebooks directory shipped with this project."
  def notebooks_path, do: @notebooks

end
