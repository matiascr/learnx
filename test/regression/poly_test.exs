defmodule PolyTest do
  use ExUnit.Case
  doctest Learnx.PolynomialRegression

  import Learnx.TestHelper
  import Nx

  alias Learnx.PolynomialRegression, as: PolyReg

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

  # Fit transform simple
  describe "transform" do
    test "polynomial regression transform | simple | bias", state do
      feature_matrix = PolyReg.transform(tensor(state.x_alt), 3, true)

      assert feature_matrix ==
               ~M[1  -1   1  -1
                  1   1   1   1
                  1   3   9  27
                  1   5  25 125]
    end

    test "polynomial regression transform | simple | no bias", state do
      feature_matrix = PolyReg.transform(tensor(state.x_alt), 3, false)

      assert feature_matrix ==
               ~M[-1   1  -1
                   1   1   1
                   3   9  27
                   5  25 125]
    end
  end

  describe "transform multiple" do
    test "polynomial regression transform | multiple | bias", state do
      feature_matrix = PolyReg.transform(tensor(state.x_multi), 3, true)

      assert feature_matrix ==
               ~M[1   1   1   1   1   1   1   1   1   1
                  1   1   2   1   2   4   1   2   4   8
                  1   2   2   4   4   4   8   8   8   8
                  1   2   3   4   6   9   8  12  18  27]
    end

    test "polynomial regression transform | multiple | no bias", state do
      feature_matrix = PolyReg.transform(tensor(state.x_multi), 3, false)

      assert feature_matrix ==
               ~M[1   1   1   1   1   1   1   1   1
                  1   2   1   2   4   1   2   4   8
                  2   2   4   4   4   8   8   8   8
                  2   3   4   6   9   8  12  18  27]
    end
  end

  describe "fit" do
    test "polynomial regression fit | bias", state do
      reg = PolyReg.fit(state.x, state.y, 3)
      assert reg.n_features == 1
      assert reg.degree == 3
      assert reg.intercept |> approx(4) == 7

      assert reg.coef |> approx(4) ==
               [1.00000000e+00, -1.99840144e-15, 3.33066907e-16] |> approx(4)
    end

    test "polynomial regression fit | no bias", state do
      reg = PolyReg.fit(state.x, state.y, 3, include_bias: false)
      assert reg.n_features == 1
      assert reg.degree == 3
      assert reg.intercept == nil
      assert reg.coef |> approx(4) == [-0.55555556, 2.94202899, -0.47342995] |> approx(4)
    end
  end

  describe "predict" do
    test "polynomial regression predict | number | bias", state do
      obs = 3

      pred =
        PolyReg.fit(state.x, state.y, 3, include_bias: true)
        |> PolyReg.predict(obs)

      assert pred |> approx(4) == 10
    end

    test "polynomial regression predict | list | bias", state do
      obs = [-1, 0, 1, 3]

      pred =
        PolyReg.fit(state.x, state.y, 3, include_bias: true)
        |> PolyReg.predict(obs)

      assert pred |> approx(4) == [6, 7, 8, 10]
    end

    test "polynomial regression predict | number | no bias", state do
      obs = 3

      pred =
        PolyReg.fit(state.x, state.y, 3, include_bias: false)
        |> PolyReg.predict(obs)

      assert pred |> approx(4) == 12.02898551 |> approx(4)
    end

    test "polynomial regression predict | list | no bias", state do
      obs = [-1, 0, 1, 3]

      pred =
        PolyReg.fit(state.x, state.y, 3, include_bias: false)
        |> PolyReg.predict(obs)

      assert pred |> approx(4) == [3.97101449, 0, 1.91304348, 12.02898551] |> approx(4)
    end
  end
end
