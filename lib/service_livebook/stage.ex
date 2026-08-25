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

  @scale_max 10.0

  @doc """
  The scale a score arrives on: 0 to #{@scale_max}.

  EditScore returns each component out of ten and `overall` as their geometric mean, so a
  score lands in 0..10 and not 0..1. That was measured -- a matching instruction scored
  4.29 and a nonsense instruction on the same pair scored 0.00 -- and it is recorded in
  `logbook-fourloops-first-runs.md`.

  IT IS PART OF THE CONTRACT RATHER THAN A REMARK BECAUSE ASSUMING IT HAS COST TWICE. A
  constant compared against an assumed range is the defect, not the constant: the router
  first compared a variance against 0.15 on a scale it thought was 0..1, then a standard
  deviation against 0.15 on a range ten times larger. Both were wrong in the same way, and
  a mock returning 0.58 for a stage that really returns 5.8 is how a test suite agrees with
  the mistake.
  """
  @spec scale() :: {float(), float()}
  def scale, do: {0.0, @scale_max}

  @doc """
  True when a number is on the scale a score is declared to arrive on.

  THE FLOOR THIS CANNOT REACH: 0.58 is a legal score and so is 5.8, so a stage that
  silently returns a 0..1 number passes this check. What it catches is the out-of-range
  case -- a percentage, a raw logit, a negative -- and the scale being stated at all is
  what makes the mocks in the test suite comparable to the thing they stand in for.
  """
  @spec on_scale?(term()) :: boolean()
  def on_scale?(value) when is_number(value), do: value >= 0.0 and value <= @scale_max
  def on_scale?(_), do: false

  @doc "Produce an artifact for round `index`, given the loop's inputs."
  @callback propose(index :: pos_integer(), input :: map()) ::
              {:ok, artifact()} | {:unavailable, String.t()} | {:error, term()}

  @doc "Score an artifact against what conditioned it. A refusal is not a zero."
  @callback score(artifact :: artifact(), input :: map()) ::
              {:ok, float()} | {:refused, String.t()} | {:error, term()}
end
