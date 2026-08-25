defmodule ServiceLivebook.ParityTest do
  @moduledoc """
  The Elixir loop and the Python loop must agree, because there are two of them.

  `ServiceLivebook.Loop` exists so the loop can be driven by a Mox mock with no card.
  `priv/python/weft_loop.py` is what the notebooks actually run. Two implementations of one
  loop drift, and nothing would report it, so this drives both with the same scripted
  sequence and compares what they produce.

  Only the invariants both claim are compared: the baseline is measured first and on the
  control, a delta is the score minus that baseline, and a target stops the run. The
  outcome vocabulary is not compared, because only the Elixir side has one -- the Python
  side raises instead, which is its own way of refusing to record a non-measurement.

  If Python is missing this fails. A skipped parity check reads exactly like a passing one.
  """
  use ExUnit.Case, async: true

  alias ServiceLivebook.Loop

  @script [0.30, 0.52, 0.61, 0.66]

  defmodule ScriptedStage do
    @moduledoc "The same script the Python side is given: control first, then each round."
    @behaviour ServiceLivebook.Stage

    @impl true
    def propose(index, _input), do: {:ok, "round_#{index}.png"}

    @impl true
    def score(artifact, %{script: script}) do
      index = if artifact == "control.png", do: 0, else: round_index(artifact)
      {:ok, Enum.at(script, index)}
    end

    defp round_index("round_" <> rest), do: rest |> String.trim_trailing(".png") |> String.to_integer()
  end

  test "both loops report the same baseline, deltas and round count" do
    {:ok, history} =
      Loop.run(ScriptedStage, %{control: "control.png", script: @script}, rounds: 3)

    elixir = %{
      "baseline" => history.baseline,
      "deltas" => Enum.map(history.rounds, &Float.round(&1.delta, 6)),
      "rounds" => length(history.rounds)
    }

    python = run_python(@script, 3, nil)

    assert python == elixir
  end

  test "both loops stop on the same round when a target is given" do
    script = [0.30, 0.52, 0.91, 0.99]

    {:ok, history} =
      Loop.run(ScriptedStage, %{control: "control.png", script: script}, rounds: 3, target: 0.90)

    python = run_python(script, 3, 0.90)

    assert python["rounds"] == length(history.rounds)
    assert python["rounds"] == 2
  end

  defp run_python(script, rounds, target) do
    dir = Path.expand("../priv/python", __DIR__)
    tmp = Path.join(System.tmp_dir!(), "parity_#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp)
    artifact = Path.join(tmp, "artifact.png")
    File.write!(artifact, "not an image, and nothing here opens it")

    program = """
    import json, sys
    sys.path.insert(0, r"#{dir}")
    from weft_loop import run

    script = #{Jason.encode!(script)}
    seen = {"n": 0}
    control = r"#{artifact}"

    def score(path):
        if path == control and seen["n"] == 0:
            seen["n"] = 1
            return script[0]
        value = script[seen["n"]]
        seen["n"] += 1
        return value

    history = run(lambda i: control, score, control=control, rounds=#{rounds},
                  target=#{if target, do: target, else: "None"})
    print(json.dumps({
        "baseline": history.baseline,
        "deltas": [round(r.delta, 6) for r in history.rounds],
        "rounds": len(history.rounds),
    }))
    """

    {out, status} = System.cmd("python", ["-c", program], stderr_to_stdout: true)
    assert status == 0, "the python harness did not run, and a skipped parity check is not a pass:\n#{out}"

    out |> String.trim() |> String.split("\n") |> List.last() |> Jason.decode!()
  end
end
