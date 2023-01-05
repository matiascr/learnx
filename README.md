# Learnx

Learnx is a pure Elixir library for Machine Learning built on top of Nx for
data and calculations and VegaLite for visualization.

The main idea is to provide established and widely used Machine Learning
and Data Processing algorithms in a familiar manner to people coming from
other languages, as well as providing comprehensive tools for newcomers.

A lot of focus is put in versatility, documentation and test-driven
development. It is important to have precision en par with other
implementations of these algorithms.

## Scope of project

The project is intended to eventually to provide implementations for:

- Data Preprocessing
- Regression
  - [x] Linear
  - [x] Polynomial
  - [ ] SVR
  - [ ] NN
  - [ ] Naive Bayes
  - [ ] Decision Trees
  - [ ] And more...
- Classification
- Clustering
- Model Selection

## Progress

Currently available functionality:

### Regression

#### Simple Linear Regression

```elixir
iex> x = [-1, 1, 3, 5]
iex> y = [6, 8, 10, 12]
iex> reg = Learnx.LinearRegression.fit(x, y)
iex> reg |> Learnx.LinearRegression.predict(2)
9.0
iex> reg |> Learnx.LinearRegression.predict([4, 6])
[11.0, 13.0]
```

#### Multiple Linear Regression

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
iex> reg |> Learnx.LinearRegression.predict([3, 5])
[15.999993324279785]
iex> reg |> Learnx.LinearRegression.predict([[3, 3], [3, 5]])
[11.999993324279785, 15.999993324279785]
```

#### Simple Polynomial Regression

```elixir
iex> x = [-1, 1, 3, 5]    
iex> y = [6, 8, 9, 11]
iex> Learnx.PolynomialRegression.transform(x, 3)
#Nx.Tensor<
  s64[4][4]
  [
    [1, -1, 1, -1],
    [1, 1, 1, 1],
    [1, 3, 9, 27],
    [1, 5, 25, 125]
  ]
>
iex> reg = Learnx.PolynomialRegression.fit(x, y, 3) 
iex> reg.coef
#Nx.Tensor<
  f32[3]
  [0.9583308696746826, -0.24999012053012848, 0.04166489839553833]
>
iex> reg.intercept
#Nx.Tensor<
  f32
  7.249996185302734
>
iex> reg |> Learnx.PolynomialRegression.predict(0)
7.249996185302734
iex> reg |> Learnx.PolynomialRegression.predict([0,2])
[7.249996185302734, 8.500016689300537]
```

## Installation

If [available in Hex](https://hex.pm/docs/publish) (not yet), the package can be installed
by adding `sphinx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:sphinx, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/sphinx>.

Independently started in 2022 by myself.
