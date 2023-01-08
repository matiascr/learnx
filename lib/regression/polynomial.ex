defmodule Learnx.PolynomialRegression do
  @moduledoc """
  Polynomial Regression.

  Generates new feature matrix consisting of all polynomial combinations
  of the features with degree less than or equal to the specified degree.
  For example, if an input sample is two dimensional and of the form [a, b],
  the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
  Then, with this feature matrix, a Linear Regression can be performed.
  """

  import Learnx.Preprocessing
  import Nx
  import Nx.LinAlg

  alias __MODULE__, as: Poly

  @typedoc """
  Polynomial regressor.

  - coef : tensor of shape {n_polynomial_features}.
  Estimated coefficients for the linear regression problem. The dimensions
  of the tensor will depend on the number of polynomial features generated
  by the features matrix, which depends on the degree and number of features.

  - intercept : float.
  Independent term in the linear model. Set to nil if fit_intercept=false.

  - n_features : int.
  Number of features seen during fit.

  - degree : positive int, default=2.
  If a single int is given, it specifies the maximal degree of the polynomial
  features. If a tuple (min_degree, max_degree) is passed, then min_degree is
  the minimum and max_degree is the maximum polynomial degree of the generated
  features. Note that min_degree=0 and min_degree=1 are equivalent as
  outputting the degree zero term is determined by include_bias.
  """
  defstruct [:coef, :intercept, :n_features, :degree]

  @type regressor :: %Poly{
          coef: tensor,
          intercept: number,
          n_features: number,
          degree: number
        }
  @type tensor :: Nx.Tensor.t()

  @doc """
  Fits polynomial model.

  ## Parameters
  - x : list or tensor of shape {n_samples, n_features} or {n_samples,}.
  Training data.

  - y : list or tensor of shape {n_samples}.
  Target values. Will be cast to X's type if necessary.

  ## Returns
  A polynomial regressor containing the coefficients, intercept (if chosen),
  number of features and the degree.

  ## Examples
  Using lists as inputs, degree 3:
  ```elixir
  iex> x = [-1, 1, 3, 5]
  iex> y = [6, 8, 10, 12]
  iex> reg = Learnx.PolynomialRegression.fit(x, y, 3)
  iex> reg.n_features
  1
  iex> reg.degree
  3
  iex> reg.intercept
  6.999995231628418
  iex> reg.coef
  #Nx.Tensor<
    f32[3]
    [0.9999974966049194, 1.0758638381958008e-5, -1.9297003746032715e-6]
  >
  ```
  # Using tensors as inputs, degree 2, no bias:
  # ```elixir
  # iex> x = [[1, 1], [1, 2], [2, 2], [2, 3]]
  # iex> y = [6, 8, 10, 12]
  # iex> reg = Learnx.PolynomialRegression.fit(x, y, 3, include_bias: true)
  # iex> reg.n_features
  # 1
  # iex> reg.degree
  # 2
  # iex> reg.intercept
  # nil
  # iex> reg.coef
  # #Nx.Tensor<
  #   f32[2]
  #   [1.5637593269348145, 0.23489943146705627]
  # >
  # ```
  """
  @spec fit(list | tensor, list | tensor, non_neg_integer, keyword) :: regressor
  def fit(x, y, degree \\ 2, opts \\ [])

  def fit(x, y, degree, opts) do
    default = [include_bias: true]
    opts = Keyword.merge(default, opts)

    {x, y} = validate_data(x, y)
    {_n_samples, n_features} = shape(x)

    coef =
      x
      |> transform(degree, opts[:include_bias])
      |> compute_coefs(y)

    case opts[:include_bias] do
      true ->
        %Poly{
          coef: coef[1..-1//1],
          intercept: coef[0] |> to_number(),
          n_features: n_features,
          degree: degree
        }

      false ->
        %Poly{
          coef: coef,
          intercept: nil,
          n_features: n_features,
          degree: degree
        }
    end
  end

  @doc """
  Transforms and returns x.

  Generates the feature matrix from the matrix, the result is based on the
  degree.

  ## Parameters
  - x : tensor of shape {n_samples, n_features}.
  Samples.

  - degree : non-negative integer.
  Degree of the transformed matrix to return.

  - include_bias : boolean, default=true.
  If `true`, then include a bias column, the feature in which all
  polynomial powers are zero (i.e. a column of ones - acts as an intercept
  term in a linear model).

  ## Returns
  Transformed version of x.
  """
  @spec transform(tensor, pos_integer, boolean) :: any
  def transform(x, n, bias \\ false) when is_tensor(x) do
    {n_samples, _n_features} = shape(x)

    res =
      for i <- 0..(n_samples - 1) do
        x[[i]]
        |> to_flat_list()
        |> matrix(1, n)
        |> List.flatten()
        |> tensor()
      end
      |> stack()

    case bias do
      true ->
        [
          broadcast(1, {1, n_samples}),
          res |> transpose()
        ]
        |> concatenate()
        |> transpose()

      false ->
        res
    end
  end

  defp matrix(l, 1, n) when is_list(l), do: [l] ++ matrix(l, l, 2, n)
  defp matrix(b, _c, m, n) when is_list(b) and m > n, do: []

  defp matrix(base, cumulative, m, n) when is_list(base) and m <= n and n > 1 do
    n_features = length(base)
    t_base = tensor(base)
    t_cumulative = tensor(cumulative)

    r =
      for c <- 0..(n_features - 1) do
        start = c * (m - 1)

        t_base[c]
        |> multiply(t_cumulative[start..-1//1])
      end
      |> Enum.map(&to_flat_list(&1))
      |> List.flatten()

    r ++ matrix(base, r, m + 1, n)
  end

  @spec compute_coefs(tensor, tensor) :: tensor
  defp compute_coefs(dm, y) do
    m = axis_size(dm, 1)
    n = axis_size(y, 0)

    if m > n,
      do: raise("This data cannot be computed. Try adding more samples. {m: #{m} > n: #{n}}")

    dm
    |> transpose()
    |> dot(dm)
    |> invert()
    |> dot(transpose(dm))
    |> dot(y)
  end

  @doc """
  Predicts using the polynomial model.

  ## Parameters
  - regressor : trained regressor to use for the predictions.

  - x : tensor of shape {n_samples, n_features}.
  Samples.

  ## Returns
  Predicted value or values. Returns the prediction(s) in the type of the
  input, i. e. if x is a tensor, it returns a tensor; and if x is a number,
  it returns a number.
  """

  @spec predict(regressor | any, number | list) :: list | number
  def predict(regressor, x) when is_list(x) do
    Enum.map(x, &predict(regressor, &1))
  end

  def predict(%Poly{coef: coef, intercept: nil, degree: degree}, x) do
    1..degree
    |> Enum.map(fn d -> x |> power(d) end)
    |> stack()
    |> multiply(coef)
    |> sum()
    |> to_number()
  end

  def predict(regressor = %Poly{intercept: intercept}, x) do
    intercept + predict(%{regressor | intercept: nil}, x)
  end
end
