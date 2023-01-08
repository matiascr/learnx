defmodule LinearTest do
  use ExUnit.Case
  doctest Learnx.LinearRegression

  import Learnx.Helper
  import Nx

  alias Learnx.LinearRegression, as: Linear

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

  describe "fit" do
    test "linear regression fit | one feature | intercept", state do
      reg = Linear.fit(state.x, state.y)
      assert reg.coef |> approx(4) == 1.0
      assert reg.intercept |> approx(4) == 7.0
      assert reg.n_features == 1
    end

    test "linear regression fit | multiple features | intercept", state do
      reg = Linear.fit(state.x_multi, state.y_alt)
      assert reg.coef |> approx(4) == [1.0, 2.0]
      assert reg.intercept |> approx(4) == 3.0
      assert reg.n_features == 2
    end

    test "linear regression fit | multiple features | no intercept", state do
      reg = Linear.fit(state.x_multi, state.y_alt, intercept: false)
      assert reg.coef |> approx(4) == [2.0909, 2.5454]
      assert reg.intercept == nil
      assert reg.n_features == 2
    end
  end

  describe "predict" do
    test "linear regression predict | one feature | number", state do
      pred =
        Linear.fit(state.x, state.y)
        |> Linear.predict(1)

      assert pred == 8.0
    end

    test "linear regression predict | one feature | list", state do
      pred =
        Linear.fit(state.x, state.y)
        |> Linear.predict([1, 3])

      assert pred == [8.0, 10.0]
    end

    test "linear regression predict | one feature | tensor", state do
      pred =
        Linear.fit(state.x, state.y)
        |> Linear.predict(tensor([1, 3]))

      assert pred |> approx(4) == [8.0, 10.0]
    end

    test "linear regression predict | multiple features | 1 sample", state do
      pred =
        Linear.fit(state.x_multi, state.y)
        |> Linear.predict([1, 1])

      assert pred |> approx(4) == [6.0]
    end

    test "linear regression predict | multiple features | list of samples", state do
      pred =
        Linear.fit(state.x_multi, state.y)
        |> Linear.predict([[1, 1], [1, 2], [2, 2]])

      assert pred |> approx(4) == [6.0, 8.0, 10.0]
    end

    test "linear regression predict | multiple features | tensor of samples", state do
      pred =
        Linear.fit(state.x_multi, state.y)
        |> Linear.predict(tensor([[1, 1], [1, 2], [2, 2]]))
        |> to_flat_list()

      assert pred |> approx(4) == [6.0, 8.0, 10.0]
    end
  end
end
