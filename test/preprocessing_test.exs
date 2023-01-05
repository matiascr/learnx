defmodule PreprocessingTest do
  use ExUnit.Case
  doctest Learnx.Preprocessing

  import Nx

  alias Learnx.Preprocessing

  setup_all do
    x = [-1, 1, 3, 5]
    x_alt = [[-1], [1], [3], [5]]
    x_multi = [[1, 1], [1, 2], [2, 2], [2, 3]]
    x_fail = [1, 2, 3, 5, 6]
    x_alt_fail = [[1], [2], [3], [5], [6]]
    x_multi_fail = [[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]]

    y = [6, 8, 10, 12]
    y_alt = [[6], [8], [9], [11]]

    {
      :ok,
      x: x,
      x_alt: x_alt,
      x_multi: x_multi,
      y: y,
      y_alt: y_alt,
      x_fail: x_fail,
      x_alt_fail: x_alt_fail,
      x_multi_fail: x_multi_fail
    }
  end

  test "single obs array is valid", state do
    assert Preprocessing.validate_data(state.x, state.y) ==
             {tensor(state.x_alt), tensor(state.y)}

    assert_raise RuntimeError, fn -> Preprocessing.validate_data(state.x_fail, state.y) end
  end

  test "nested obs array is valid", state do
    assert Preprocessing.validate_data(state.x_alt, state.y) ==
             {tensor(state.x_alt), tensor(state.y)}

    assert_raise RuntimeError, fn -> Preprocessing.validate_data(state.x_alt_fail, state.y) end
  end

  test "multiple obs array is valid", state do
    assert Preprocessing.validate_data(state.x_multi, state.y) ==
             {tensor(state.x_multi), tensor(state.y)}

    assert_raise RuntimeError, fn -> Preprocessing.validate_data(state.x_multi_fail, state.y) end
  end
end
