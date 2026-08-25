# Gate: no exception handling in this project's Elixir.
#
# WHY THIS EXISTS. Armstrong's thesis and the Erlang book it came from put the recovery in
# the supervisor rather than in the code that failed: a process that meets something it was
# not written for should die and be restarted by something whose job is deciding what to do
# about that. A `try/rescue` around the failure moves that decision into the worker, where
# it is made once, badly, by whoever was in a hurry -- and the shape it usually takes is a
# rescue that logs and returns `nil`, which is the same defect this workspace keeps writing
# down: a silent skip reads exactly like a pass.
#
#   https://erlang.org/download/erlang-book-part1.pdf
#
# WHAT IS FORBIDDEN. `try`, `rescue`, `catch`, `after` and `throw`. These are the
# constructs that stop a crash from reaching a supervisor.
#
# WHAT IS NOT, AND THIS IS AN INTERPRETATION SOMEBODY MAY WANT TO OVERRULE. `raise` and the
# bang functions -- `Map.fetch!`, `File.read!` -- are permitted. Crashing IS the mechanism
# let-it-crash asks for, so a gate that banned raising would ban the philosophy it is named
# after; `Map.fetch!(input, :control)` in `ServiceLivebook.Loop` is the intended behaviour
# for a loop started without a control, not a defect. What breaks the model is catching.
# If the rule was meant as "no raising either", this file is where that changes, and the
# controls below say what would have to change with it.
#
# WHY IT PARSES RATHER THAN GREPS. `grep rescue` finds the word in this comment, in a
# docstring, and in a string literal -- the convenient proxy rather than the quantity. This
# reads the AST, and one of the controls is a file whose only `rescue` is inside a comment
# and a string, which must PASS.
#
# PARSING GOES THROUGH SOURCEROR. `Code.string_to_quoted/2` also parses, and the first
# version used it; what Sourceror adds is the range of the construct -- start and end, line
# and column -- so a violation is reported at the `rescue` rather than at the `def` that
# encloses it, and so this gate can be turned into a fixer without being rewritten, because
# Sourceror patches source back out without reformatting the file around it.
#
# It is a dependency where there was none. `mix deps.get` now has to run before this gate
# does, which is a line in the workflow and a reason the hook is not standalone any more.
#
# WHAT IS NOT COVERED, named rather than omitted. This gate reads `lib/` and `test/` in this
# project only. A sweep of the other Elixir checked out beside it finds `try` or `rescue` in
# 82 files under `1-transport/ex_mcp` and 24 under `3-interactor/nx-shuttle`. Neither is
# touched here: adopting the rule there is a change to those projects, made by whoever
# maintains them, and pretending this gate covers them would be worse than saying it does
# not.
#
#     mix run --no-start scripts/check_no_exceptions.exs [path ...]
#     mix run --no-start scripts/check_no_exceptions.exs --self-test
#
# Exit code is non-zero on any violation, and on any control that fails to fail.

defmodule CheckNoExceptions do
  @why %{
    try: "try/rescue keeps a crash from reaching the supervisor that would restart it",
    throw: "throw is control flow through the exception system",
    rescue: "a rescue clause catches what a supervisor should have been told about",
    catch: "a catch clause swallows an exit the supervision tree is there to handle",
    after: "an after clause on a function body is the tail of a try, however it is spelled"
  }

  # `def foo do ... rescue ... end` IS a try, and it does not produce a `:try` node -- the
  # clauses arrive as a keyword list on the def. The first version of this gate walked for
  # `:try` alone, and the most idiomatic way to write a rescue in Elixir went straight
  # through it: a control file with a function-level rescue was reported as clean. A gate
  # that passes on known-broken input certifies the defect, so both shapes are read.
  @definitions [:def, :defp, :defmacro, :defmacrop]

  # `after` on a `receive` is a timeout and nothing to do with exceptions. It is left alone
  # by reading clause keywords only where they hang off a definition or a try, and `receive`
  # is neither.
  @clause_keys [:rescue, :catch, :after]

  @doc "Every violation in one source, as {line, column, construct}."
  def scan(source, name \\ "nofile") do
    case Sourceror.parse_string(source) do
      {:ok, ast} ->
        {_ast, found} = Macro.prewalk(ast, [], &visit/2)
        {:ok, found |> Enum.uniq() |> Enum.sort()}

      {:error, reason} ->
        {:parse_error, "#{name}: #{inspect(reason)}"}
    end
  end

  defp visit({:try, _meta, args} = node, acc) when is_list(args) do
    {node, [at(node, :try) | acc]}
  end

  defp visit({:throw, _meta, args} = node, acc) when is_list(args) do
    {node, [at(node, :throw) | acc]}
  end

  defp visit({definition, _meta, args} = node, acc)
       when definition in @definitions and is_list(args) do
    {node, clauses(args, node) ++ acc}
  end

  defp visit(node, acc), do: {node, acc}

  # SOURCEROR WRAPS LITERALS, AND THAT SILENTLY UNDID THIS CHECK. Where
  # `Code.string_to_quoted/2` gives a def's clauses as a plain keyword list -- `[do: ...,
  # rescue: ...]` -- Sourceror encodes every literal as `{:__block__, meta, [literal]}`, so
  # the keys are nodes and `Keyword.keyword?/1` answers false. Swapping the parser under an
  # unchanged `Keyword.keys/1` turned the def-rescue check off and reported the control file
  # as clean, which is the second time that same file has caught this gate. Keys are read
  # through `key_of/1` now, in either encoding.
  defp clauses(args, definition) do
    args
    |> Enum.filter(&is_list/1)
    |> Enum.flat_map(fn kw ->
      kw
      |> Enum.filter(&match?({_key, _value}, &1))
      |> Enum.filter(fn {key, _value} -> key_of(key) in @clause_keys end)
      |> Enum.map(fn {key, value} -> at(value, key_of(key), definition) end)
    end)
  end

  defp key_of({:__block__, _meta, [key]}) when is_atom(key), do: key
  defp key_of(key) when is_atom(key), do: key
  defp key_of(_), do: nil

  # The range of the construct, falling back to the node that encloses it. A clause keyword
  # carries no range of its own, so what is reported is the range of its VALUE -- the clause
  # body, which is the line below the `rescue` keyword itself. That is close enough to be
  # the line a reader edits, and it beats naming the enclosing `def`, which can be forty
  # lines up.
  defp at(node, construct, fallback \\ nil) do
    case range(node) || range(fallback) do
      %{start: [line: line, column: column]} -> {line, column, construct}
      _ -> {0, 0, construct}
    end
  end

  defp range(nil), do: nil

  defp range(node) do
    Sourceror.get_range(node)
  end

  @doc "Violations across a list of files, as printable lines."
  def check(paths) do
    Enum.flat_map(paths, fn path ->
      case scan(File.read!(path), path) do
        {:ok, []} ->
          []

        {:ok, found} ->
          Enum.map(found, fn {line, column, construct} ->
            "#{path}:#{line}:#{column}: #{construct} -- #{@why[construct]}"
          end)

        {:parse_error, message} ->
          ["#{path}: does not parse, so it was not checked -- #{message}"]
      end
    end)
  end

  @doc "Every .ex and .exs this project owns. deps/ and _build/ are other people's."
  def sources(roots) do
    roots
    |> Enum.flat_map(&Path.wildcard(Path.join(&1, "**/*.{ex,exs}")))
    |> Enum.reject(&String.contains?(&1, ["/deps/", "/_build/", "/.pixi/"]))
    |> Enum.sort()
  end

  @clean """
  defmodule Clean do
    @moduledoc "A rescue in a comment, and a rescue in a string. Neither is a rescue."

    # A grep for rescue finds this line, and it is prose.
    def describe, do: "the word rescue, in a string, inside a function that catches nothing"

    def run(stage, input) do
      case stage.score(input) do
        {:ok, value} -> {:ok, value}
        {:refused, why} -> {:refused, why}
      end
    end
  end
  """

  @rescued """
  defmodule Rescued do
    def run(stage, input) do
      try do
        stage.score(input)
      rescue
        error -> {:error, error}
      end
    end
  end
  """

  @caught """
  defmodule Caught do
    def run(stage, input) do
      try do
        stage.score(input)
      catch
        :exit, reason -> {:error, reason}
      end
    end
  end
  """

  @after_block """
  defmodule Afterwards do
    def run(stage, input) do
      try do
        stage.score(input)
      after
        :ok
      end
    end
  end
  """

  @thrown """
  defmodule Thrown do
    def run(scores) do
      Enum.each(scores, fn score -> if score > 10, do: throw({:off_scale, score}) end)
    end
  end
  """

  @def_rescue """
  defmodule DefRescue do
    def run(stage, input) do
      stage.score(input)
    rescue
      error -> {:error, error}
    end
  end
  """

  @receive_after """
  defmodule Recv do
    def run do
      receive do
        msg -> msg
      after
        100 -> :timeout
      end
    end
  end
  """

  @raising """
  defmodule Raising do
    def run(input), do: Map.fetch!(input, :control)
  end
  """

  def self_test do
    cases = [
      {"a module that catches nothing, with the word in a comment and a string", @clean, false},
      {"try/rescue", @rescued, true},
      {"try/catch", @caught, true},
      {"try/after", @after_block, true},
      {"throw as control flow", @thrown, true},
      {"a rescue on the function body rather than a try block", @def_rescue, true},
      {"receive/after, which is a timeout and not a rescue", @receive_after, false},
      {"raise and bang functions, which are permitted on purpose", @raising, false}
    ]

    IO.puts("self-test: each known-bad input must be rejected, and prose must not be")

    results =
      Enum.map(cases, fn {label, source, should_fail} ->
        {:ok, found} = scan(source, "control.ex")
        failed = found != []
        good = failed == should_fail
        detail =
          if failed do
            " " <> inspect(Enum.map(found, fn {line, column, construct} -> "#{construct}@#{line}:#{column}" end))
          else
            ""
          end
        IO.puts("  #{if good, do: "ok ", else: "BAD"} #{label}: #{if failed, do: "rejected", else: "accepted"}#{detail}")
        good
      end)

    if Enum.all?(results), do: 0, else: 1
  end

  def main(argv) do
    if "--self-test" in argv do
      System.halt(self_test())
    end

    roots = if argv == [], do: ["lib", "test", "scripts"], else: argv
    files = sources(roots)
    problems = check(files)
    Enum.each(problems, &IO.puts/1)
    IO.puts("#{length(files)} file(s) read, #{length(problems)} violation(s)")
    System.halt(if problems == [], do: 0, else: 1)
  end
end

CheckNoExceptions.main(System.argv())
