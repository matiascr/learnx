defmodule LearnxTest do
  use ExUnit.Case
  doctest Learnx.Helper

  import Nx
  import Learnx.Helper

  test "sparse" do
    t =
      tensor([
        [1, 2, 3],
        [2, 3, 4]
      ])

    assert sparse?(t) == false

    t =
      tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ])

    assert sparse?(t) == true
  end
end
