defmodule Learnx.LinearRegression do
  @moduledoc """
  Ordinary least squares Linear Regression.

  LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
  to minimize the residual sum of squares between the observed targets in
  the dataset, and the targets predicted by the linear approximation.
  """

  import Learnx.Preprocessing
  import Nx
  import Nx.LinAlg
  import Nx.Defn

  alias __MODULE__, as: Linear

  @typedoc """
  Linear regressor.

  - coef : tensor of shape {n_features, }.
  Estimated coefficients for the linear regression problem.

  - intercept : float.
  Independent term in the linear model. Set to nil if fit_intercept = False.

  - n_features : int.
  Number of features seen during fit.
  """
  defstruct [:coef, :intercept, :n_features]

  @type regressor :: %Linear{
          coef: tensor,
          intercept: number,
          n_features: number
        }
  @type tensor :: Nx.Tensor.t()

  @doc """
  Fits linear model.

  ## Parameters
  - x : list or tensor of shape {n_samples, n_features} or {n_samples,}.
  Training data.

  - y : list or tensor of shape {n_samples,}.
  Target values. Will be cast to X's type if necessary.

  - intercept : boolean.
  Wether or not to use intercept in calculations. Simple linear regression
  will always use intercept.

  ## Returns
  A linear regressor containing the coefficients, intercept (if chosen) and
  number of features.

  ## Examples
  With lists as inputs:
  ```elixir
  iex> x = [-1, 1, 3, 5]
  iex> y = [6, 8, 10, 12]
  iex> reg = Learnx.LinearRegression.fit(x, y)
  iex> reg.coef
  #Nx.Tensor<
    f32
    1.0
  >
  iex> reg.intercept
  7.0
  iex> reg.n_features
  1
  ```

  With tensors:
  ```elixir
  iex> x = Nx.tensor([-1, 1, 3, 5])
  iex> y = Nx.tensor([6, 8, 10, 12])
  iex> reg = Learnx.LinearRegression.fit(x, y)
  iex> reg.coef
  #Nx.Tensor<
    f32
    1.0
  >
  iex> reg.intercept
  7.0
  iex> reg.n_features
  1
  ```

  With multiple features as the input:
  ```elixir
  iex> x = [[1, 1], [1, 2], [2, 2], [2, 3]]
  iex> y = [6, 8, 9, 11]
  iex> reg = Learnx.LinearRegression.fit(x, y)
  iex> reg.coef
  #Nx.Tensor<
    f32[2]
    [1.0, 2.0]
  >
  iex> reg.intercept
  2.999993324279785
  iex> reg.n_features
  2
  ```
  """
  @spec fit(list | tensor, list | tensor, keyword) :: regressor
  def fit(x, y, opts \\ []) do
    default = [intercept: true]
    opts = Keyword.merge(default, opts)

    validate_data(x, y)
    |> compute(opts[:intercept])
  end

  @spec compute({tensor, tensor}, boolean) :: regressor
  defp compute({x, y}, intercept) do
    {_n_samples, n_features} = shape(x)

    case {n_features == 1, intercept} do
      {true, _} ->
        coef = solve_simple(x, y)

        %Linear{
          coef: coef[1],
          intercept: coef[0] |> to_number(),
          n_features: n_features
        }

      {false, true} ->
        coef = solve_multiple(x, y, true)

        %Linear{
          coef: coef[1..-1//1],
          intercept: coef[0] |> to_number(),
          n_features: n_features
        }

      {false, false} ->
        coef = solve_multiple(x, y, false)

        %Linear{
          coef: coef,
          n_features: n_features,
          intercept: nil
        }
    end
  end

  @spec solve_simple(tensor, tensor) :: regressor
  defnp solve_simple(x, y) do
    x = x |> squeeze()

    b_1 = ss(x, y) / ss(x, x)
    b_0 = mean(y) - b_1 * mean(x)
    stack([b_0, b_1])
  end

  @spec ss(tensor, tensor) :: tensor
  defnp ss(x, y) do
    sum((x - mean(x)) * (y - mean(y)))
  end

  @spec solve_multiple(tensor, tensor, boolean) :: regressor
  defp solve_multiple(x, y, false) do
    x
    |> transpose()
    |> dot(x)
    |> invert()
    |> dot(transpose(x))
    |> dot(y)
  end

  defp solve_multiple(x, y, true) do
    {n_samples, _n_features} = shape(x)

    [
      broadcast(1, {1, n_samples})
      | x |> transpose() |> to_batched(1) |> Enum.to_list()
    ]
    |> stack()
    |> squeeze()
    |> transpose()
    |> solve_multiple(y, false)
  end

  @doc """
  Predicts using the linear model.

  ## Parameters
  - regressor : trained regressor to use for the predictions.

  - x : number, list or tensor of shape {n_observations, n_features}.
  Observations to predict.

  ## Returns
  Predicted value or values. Returns the prediction(s) in the type of the
  input, i. e. if x is a tensor, it returns a tensor; and if x is a number,
  it returns a number.

  ## Examples
  Single feature regression:
  ```elixir
  iex> x = [-1, 1, 3, 5]
  iex> y = [6, 8, 10, 12]
  iex> reg = Learnx.LinearRegression.fit(x, y)
  iex> reg |> Learnx.LinearRegression.predict(2)
  9.0
  iex> reg |> Learnx.LinearRegression.predict([4, 6])
  [11.0, 13.0]
  ```

  Multiple features regression:
  ```elixir
  iex> x = [[1, 1], [1, 2], [2, 2], [2, 3]]
  iex> y = [6, 8, 10, 12]
  iex> reg = Learnx.LinearRegression.fit(x, y)
  iex> reg |> Learnx.LinearRegression.predict([3, 3])
  [13.999992370605469]
  ```
  """
  @spec predict(regressor, number | list | tensor) :: number | list | tensor
  def predict(regressor = %Linear{n_features: 1}, x) when is_number(x),
    do: predict(regressor, tensor(x)) |> to_number()

  def predict(regressor = %Linear{n_features: 1}, x) when is_list(x),
    do: predict(regressor, tensor(x)) |> to_flat_list()

  def predict(%Linear{coef: coef, intercept: intercept, n_features: 1}, x) do
    x |> multiply(coef) |> add(intercept)
  end

  def predict(regressor = %Linear{n_features: n}, x) when is_list(x),
    do: predict(regressor, tensor(x) |> reshape({:auto, n})) |> to_flat_list()

  def predict(%Linear{coef: coef, intercept: intercept}, x) do
    x
    |> multiply(coef)
    |> sum(axes: [1])
    |> add(intercept)
  end
end
