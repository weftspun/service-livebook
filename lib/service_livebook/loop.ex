defmodule ServiceLivebook.Loop do
  @moduledoc """
  Propose, score against the control, repair, repeat.

  This is the same loop `priv/python/weft_loop.py` runs in the notebooks, expressed against
  `ServiceLivebook.Stage` so that it can be driven by a Mox mock with no card present. The
  two are checked against each other by `test/parity_test.exs`, which runs a scripted
  sequence through both and compares the histories, because two implementations of one loop
  will otherwise drift and nothing would report it.

  Three answers are kept apart on purpose, and each of them is a failure mode that would
  otherwise read as a low score.

  * `{:refused, why}` from a scorer produces no measurement. A refusal entered as a zero
    would drag an average down and look like a bad round rather than an absent one.
  * `{:unavailable, why}` from a proposer records that a stage was selected and could not
    run. Loop 4's latent arm answers this, and substituting the other arm would record a
    geometry failure as an appearance failure that was repaired.
  * A baseline that cannot be measured stops the run. A score with nothing to compare it
    against is not a measurement of anything.
  """

  defmodule Round do
    @moduledoc "One round: what it produced, what that scored, and how far above the floor."
    defstruct [:index, :artifact, :score, :delta, :outcome, :detail]

    @type t :: %__MODULE__{
            index: pos_integer(),
            artifact: String.t() | nil,
            score: float() | nil,
            delta: float() | nil,
            outcome: :scored | :refused | :unavailable,
            detail: String.t() | nil
          }
  end

  defmodule History do
    @moduledoc "A run: its floor, and every round measured against it."
    defstruct [:baseline, :control, rounds: []]

    @type t :: %__MODULE__{baseline: float(), control: String.t(), rounds: [Round.t()]}

    @doc "The best round that produced a measurement, or nil when none did."
    def best(%__MODULE__{rounds: rounds}) do
      rounds
      |> Enum.filter(&(&1.outcome == :scored))
      |> Enum.max_by(& &1.score, fn -> nil end)
    end

    @doc "Rounds that produced no measurement, by outcome. Named and counted, never omitted."
    def unmeasured(%__MODULE__{rounds: rounds}) do
      rounds |> Enum.reject(&(&1.outcome == :scored)) |> Enum.frequencies_by(& &1.outcome)
    end
  end

  @doc """
  Run `rounds` of `stage`, measuring the baseline on `input.control` first.

  Returns `{:ok, history}`, or `{:error, reason}` when the floor itself could not be
  measured. `:target` stops early on the first round that reaches it.
  """
  @spec run(module(), map(), keyword()) :: {:ok, History.t()} | {:error, term()}
  def run(stage, input, opts \\ []) do
    rounds = Keyword.get(opts, :rounds, 3)
    target = Keyword.get(opts, :target)
    control = Map.fetch!(input, :control)

    case stage.score(control, input) do
      {:ok, baseline} ->
        history = %History{baseline: baseline, control: control}
        {:ok, take(stage, input, 1, rounds, target, history)}

      {:refused, why} ->
        {:error, {:baseline_refused, why}}

      {:error, reason} ->
        {:error, {:baseline_failed, reason}}
    end
  end

  defp take(_stage, _input, index, rounds, _target, history) when index > rounds, do: history

  defp take(stage, input, index, rounds, target, history) do
    round = one(stage, input, index, history.baseline)
    history = %{history | rounds: history.rounds ++ [round]}

    cond do
      round.outcome == :scored and target != nil and round.score >= target -> history
      true -> take(stage, input, index + 1, rounds, target, history)
    end
  end

  defp one(stage, input, index, baseline) do
    case stage.propose(index, input) do
      {:ok, artifact} ->
        case stage.score(artifact, input) do
          {:ok, score} ->
            %Round{
              index: index,
              artifact: artifact,
              score: score,
              delta: score - baseline,
              outcome: :scored
            }

          {:refused, why} ->
            %Round{index: index, artifact: artifact, outcome: :refused, detail: why}

          {:error, reason} ->
            %Round{
              index: index,
              artifact: artifact,
              outcome: :refused,
              detail: inspect(reason)
            }
        end

      {:unavailable, why} ->
        %Round{index: index, outcome: :unavailable, detail: why}

      {:error, reason} ->
        %Round{index: index, outcome: :unavailable, detail: inspect(reason)}
    end
  end

  @doc """
  Which repair arm loop 4 takes, from the spread across views rather than one number.

  High variance means the views disagree about the shape, and a shape wrong along one axis
  scores well from the view that hides it. Low mean with agreement is an appearance
  failure. A single hand-picked front view cannot separate the two, which is why the views
  come from the camera sequence.

  THE THRESHOLD IS A SPREAD IN SCORE UNITS, NOT A VARIANCE, and the correction is worth
  keeping. The first version compared variance against 0.15. A score bounded in 0..1 has a
  variance of at most 0.25, reached only when half the views score 0 and half score 1, so
  that threshold fired almost never: four views at 0.80, 0.20, 0.75 and 0.15, about as
  disagreeing as views realistically get, have a variance of 0.091 and would have been
  routed to the 2D arm. A unit test caught it before any run did.

  The number itself is still unmeasured. 0.15 of standard deviation separates the two
  scripted cases, and nothing has calibrated it against real EditScore output.
  """
  @spec route([float()], float()) :: {:latent | :view, String.t()}
  def route(view_scores, spread_threshold \\ 0.15)

  def route([], _threshold), do: {:view, "no views were scored, so nothing disagrees yet"}

  def route(view_scores, threshold) do
    if :math.sqrt(variance(view_scores)) >= threshold do
      {:latent, "the views disagree about the shape"}
    else
      {:view, "the views agree and the appearance is wrong"}
    end
  end

  @doc "Population variance, the quantity `route/2` decides on."
  def variance(scores) do
    mean = Enum.sum(scores) / length(scores)
    Enum.sum(Enum.map(scores, &((&1 - mean) * (&1 - mean)))) / length(scores)
  end
end
