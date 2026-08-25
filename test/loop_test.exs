defmodule ServiceLivebook.LoopTest do
  @moduledoc """
  The four loops, run with no card present.

  Every stage is a Mox mock of `ServiceLivebook.Stage`, so what is exercised is the loop:
  the baseline, the deltas, the early stop, and the four answers a stage can give. The
  models are not exercised and this file does not pretend otherwise. `verify_on_exit!` is
  what makes an expectation a check rather than a suggestion: a stage the loop failed to
  call fails the test.

  THE NUMBERS THE MOCKS RETURN ARE ON THE SCALE THE MODELS RETURN, 0..10, and they used to
  be on 0..1. A mock is a claim about the shape of the thing it stands in for, and this one
  was making the wrong claim quietly: every assertion here passed, and the router built on
  those numbers was calibrated against a scale EditScore does not use. Nothing in a mock
  fails when it lies about a range, which is why the range is now stated in
  `ServiceLivebook.Stage` and checked by the loop.

  What this cannot catch is a stage returning 0.58 where it means 5.8. Both are legal
  scores. The scale being declared is what makes the two comparable at all.
  """
  use ExUnit.Case, async: true
  import Mox

  alias ServiceLivebook.Loop
  alias ServiceLivebook.Loop.History
  alias ServiceLivebook.Stage
  alias ServiceLivebook.StageMock

  setup :verify_on_exit!

  defp input(extra \\ %{}), do: Map.merge(%{control: "control.png"}, extra)

  describe "loop 1: keypoints, ANNY fit, render, scored against the photograph" do
    test "three rounds rise above the floor and the referee is a separate answer" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 4.0} end)

      for {i, score} <- [{1, 5.2}, {2, 6.1}, {3, 6.6}] do
        expect(StageMock, :propose, fn ^i, _ -> {:ok, "fit_#{i}.png"} end)
        expect(StageMock, :score, fn _, _ -> {:ok, score} end)
      end

      {:ok, history} = Loop.run(StageMock, input(), rounds: 3)

      assert history.baseline == 4.0
      assert Enum.map(history.rounds, & &1.index) == [1, 2, 3]
      assert Enum.all?(history.rounds, &(&1.delta > 0))
      assert History.best(history).index == 3
      assert History.unmeasured(history) == %{}
    end
  end

  describe "loop 2: an OmniGen2 edit scored against its source" do
    test "a refusal is not a zero, and does not enter the best round" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 3.0} end)

      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, 5.8} end)

      expect(StageMock, :propose, fn 2, _ -> {:ok, "round_2.png"} end)
      expect(StageMock, :score, fn "round_2.png", _ -> {:refused, "the grader returned no number"} end)

      expect(StageMock, :propose, fn 3, _ -> {:ok, "round_3.png"} end)
      expect(StageMock, :score, fn "round_3.png", _ -> {:ok, 5.5} end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 3)

      refused = Enum.at(history.rounds, 1)
      assert refused.outcome == :refused
      assert refused.score == nil
      assert refused.delta == nil
      assert History.best(history).index == 1
      assert History.unmeasured(history) == %{refused: 1}
    end

    test "a target stops the loop rather than spending the remaining rounds" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 3.0} end)
      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, 9.1} end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 5, target: 9.0)

      assert length(history.rounds) == 1
    end

    test "the measured pair: a nonsense instruction scores zero, and zero is a measurement" do
      # The two EditScore calls that have actually run, from
      # logbook-fourloops-first-runs.md: a matching instruction scored 4.29 and a nonsense
      # instruction on the same pair scored 0.00. The zero is the negative control for the
      # scorer, and it is a measurement -- unlike a refusal, which produces no tuple.
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 4.29} end)
      expect(StageMock, :propose, fn 1, _ -> {:ok, "nonsense.png"} end)
      expect(StageMock, :score, fn "nonsense.png", _ -> {:ok, 0.00} end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 1)

      round = hd(history.rounds)
      assert round.outcome == :scored
      assert round.score == 0.0
      assert_in_delta round.delta, -4.29, 1.0e-9
      assert History.unmeasured(history) == %{}
    end
  end

  describe "loop 3: CycleGAN in front, scored against the stylized source" do
    test "the control is the stylized image, not the original" do
      stylized = "stylized_ukiyoe.png"
      expect(StageMock, :score, fn ^stylized, _ -> {:ok, 4.4} end)
      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, 6.2} end)

      {:ok, history} = Loop.run(StageMock, input(%{control: stylized}), rounds: 1)

      assert history.control == stylized
      assert history.baseline == 4.4
      assert_in_delta History.best(history).delta, 1.8, 1.0e-9
    end
  end

  describe "loop 4: the router, on both branches" do
    test "views that disagree select the latent arm" do
      assert {:latent, _} = Loop.route([8.0, 2.0, 7.5, 1.5])
    end

    test "views that agree select the 2D arm" do
      assert {:view, _} = Loop.route([4.1, 4.4, 4.0, 4.3])
    end

    test "the same views on either scale select the same arm" do
      # The control for both retracted forms of this router. Under the variance rule the
      # disagreeing set routed to the 2D arm; under the standard-deviation rule the
      # agreeing set at 0..10 routes to the latent arm. Spread over mean is dimensionless,
      # so multiplying every view by ten changes nothing, and that is the property rather
      # than a pair of hand-picked cases.
      for views <- [[0.80, 0.20, 0.75, 0.15], [0.41, 0.44, 0.40, 0.43], [0.5, 0.5, 0.5, 0.5]] do
        scaled = Enum.map(views, &(&1 * 10))
        {arm, _} = Loop.route(views)
        {scaled_arm, _} = Loop.route(scaled)
        assert arm == scaled_arm, "#{inspect(views)} routed #{arm} and #{inspect(scaled)} routed #{scaled_arm}"
      end
    end

    test "the standard deviation alone would have routed the agreeing views wrongly" do
      # Why the statistic changed, as an assertion rather than a paragraph. At 0..10 the
      # agreeing views have a spread of 0.158, above the 0.15 the old rule compared
      # directly, so that rule sends an appearance failure to the latent arm.
      agreeing = [4.1, 4.4, 4.0, 4.3]
      assert :math.sqrt(Loop.variance(agreeing)) >= 0.15
      assert {:view, _} = Loop.route(agreeing)
    end

    test "every view scoring zero is not disagreement" do
      assert {:view, why} = Loop.route([0.0, 0.0, 0.0])
      assert why =~ "zero"
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

  describe "the scale" do
    test "a score off the scale is not a measurement, and not a zero either" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 4.0} end)
      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, 58.0} end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 1)

      round = hd(history.rounds)
      assert round.outcome == :off_scale
      assert round.score == nil
      assert round.delta == nil
      assert round.detail =~ "0.0..10.0"
      assert History.best(history) == nil
      assert History.unmeasured(history) == %{off_scale: 1}
    end

    test "a negative score is off the scale as well" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 4.0} end)
      expect(StageMock, :propose, fn 1, _ -> {:ok, "round_1.png"} end)
      expect(StageMock, :score, fn "round_1.png", _ -> {:ok, -0.5} end)

      {:ok, history} = Loop.run(StageMock, input(), rounds: 1)

      assert hd(history.rounds).outcome == :off_scale
    end

    test "a baseline off the scale stops the run rather than becoming a floor" do
      expect(StageMock, :score, fn "control.png", _ -> {:ok, 87.0} end)

      assert {:error, {:baseline_off_scale, 87.0, scale}} = Loop.run(StageMock, input())
      assert scale == Stage.scale()
    end

    test "the boundary values are on the scale" do
      assert Stage.on_scale?(0.0)
      assert Stage.on_scale?(10.0)
      refute Stage.on_scale?(10.1)
      refute Stage.on_scale?(:nan)
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
