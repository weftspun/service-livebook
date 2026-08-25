defmodule ServiceLivebook.Stage do
  @moduledoc """
  The boundary every loop stage sits behind.

  A stage is whatever produces an artifact or a number: an OmniGen2 edit, an EditScore
  call, a Pixal3D generation, a VoxHammer edit. All of them are a subprocess into a `pixi`
  environment or an HTTP call to a service on this desk, and all of them need a card.

  This behaviour exists so a loop can be run without one. `ServiceLivebook.Loop` calls a
  module rather than a command, and a test supplies a mock through Mox, so the loop's own
  behaviour is exercised on its own rather than being inferred from a run that also loaded
  seventeen gigabytes of weights.

  `unavailable` is not an error and not a low score. It is the third answer a stage can
  give: the stage exists, was selected, and cannot run. VoxHammer answers this today.
  """

  @typedoc "A content-addressed artifact path, or the reason no artifact exists."
  @type artifact :: String.t()

  @doc "Produce an artifact for round `index`, given the loop's inputs."
  @callback propose(index :: pos_integer(), input :: map()) ::
              {:ok, artifact()} | {:unavailable, String.t()} | {:error, term()}

  @doc "Score an artifact against what conditioned it. A refusal is not a zero."
  @callback score(artifact :: artifact(), input :: map()) ::
              {:ok, float()} | {:refused, String.t()} | {:error, term()}
end
