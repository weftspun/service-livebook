defmodule ServiceLivebook.LoopTest do
  @moduledoc """
  The four loops, run with no card present.

  Every stage is a Mox mock of `ServiceLivebook.Stage`, so what is exercised is the loop:
  the baseline, the deltas, the early stop, and the three answers a stage can give. The
  models are not exercised and this file does not pretend otherwise. `verify_on_exit!` is
  what makes an expectation a check rather than a suggestion: a stage the loop failed to
  call fails the test.
  """
  use ExUnit.Case, async: true
  import Mox

  alias ServiceLivebook.Loop
  alias ServiceLivebook.Loop.History
  alias ServiceLivebook.StageMock

  setup :verify_on_exit!

  defp input(extra \\ %{}), do: Map.merge(%{control: "control.png"}, extra)

  describe "loop 1: keypoints, ANNY fit, render, scored against the photograph" do
    test "three rounds rise above the floor and the referee is a separate answer" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 0.40} end)

      for {i, score} <- [{1, 0.52}, {2, 0.61}, {3, 0.66}] do
        expect(StageMock, :propose, fn ^i, _ -> {:ok, "fit_#{i}.png"} end)
        expect(StageMock, :score, fn _, _ -> {:ok, score} end)
      end

      {:ok, history} = Loop.run(StageMock, input(), rounds: 3)

      assert history.baseline == 0.40
      assert Enum.map(history.rounds, & &1.index) == [1, 2, 3]
      assert Enum.all?(history.rounds, &(&1.delta > 0))
      assert History.best(history).index == 3
      assert History.unmeasured(history) == %{}
    end
  end

  describe "loop 2: an OmniGen2 edit scored against its source" do
    test "a refusal is not a zero, and does not enter the best round" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 0.30} end)

      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, 0.58} end)

      expect(StageMock, :propose, fn 2, _ -> {:ok, "round_2.png"} end)
      expect(StageMock, :score, fn "round_2.png", _ -> {:refused, "the grader returned no number"} end)

      expect(StageMock, :propose, fn 3, _ -> {:ok, "round_3.png"} end)
      expect(StageMock, :score, fn "round_3.png", _ -> {:ok, 0.55} end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 3)

      refused = Enum.at(history.rounds, 1)
      assert refused.outcome == :refused
      assert refused.score == nil
      assert refused.delta == nil
      assert History.best(history).index == 1
      assert History.unmeasured(history) == %{refused: 1}
    end

    test "a target stops the loop rather than spending the remaining rounds" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 0.30} end)
      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, 0.91} end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 5, target: 0.90)

      assert length(history.rounds) == 1
    end
  end

  describe "loop 3: CycleGAN in front, scored against the stylized source" do
    test "the control is the stylized image, not the original" do
      stylized = "stylized_ukiyoe.png"
      expect(StageMock, :score, fn ^stylized, _ -> {:ok, 0.44} end)
      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, 0.62} end)

      {:ok, history} = Loop.run(StageMock, input(%{control: stylized}), rounds: 1)

      assert history.control == stylized
      assert history.baseline == 0.44
      assert_in_delta History.best(history).delta, 0.18, 1.0e-9
    end
  end

  describe "loop 4: the router, on both branches" do
    test "views that disagree select the latent arm" do
      assert {:latent, _} = Loop.route([0.80, 0.20, 0.75, 0.15])
    end

    test "views that agree select the 2D arm" do
      assert {:view, _} = Loop.route([0.41, 0.44, 0.40, 0.43])
    end

    test "an unavailable arm is recorded, and the other arm is not silently used" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 0.35} end)

      expect(StageMock, :propose, fn 1, _ ->
        {:unavailable, "VoxHammer raises NotImplementedError outside stub mode"}
      end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 1)

      round = hd(history.rounds)
      assert round.outcome == :unavailable
      assert round.artifact == nil
      assert round.detail =~ "NotImplementedError"
      assert History.best(history) == nil
      assert History.unmeasured(history) == %{unavailable: 1}
    end
  end

  describe "the floor" do
    test "a run whose baseline cannot be scored produces no history at all" do
      expect(StageMock, :score, fn "control.png", _ -> {:refused, "no scorer"} end)

      assert {:error, {:baseline_refused, "no scorer"}} = Loop.run(StageMock, input())
    end

    test "a baseline that errors is a failure, not an empty run" do
      expect(StageMock, :score, fn "control.png", _ -> {:error, :enoent} end)

      assert {:error, {:baseline_failed, :enoent}} = Loop.run(StageMock, input())
    end
  end
end
