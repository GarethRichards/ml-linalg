# C++26: Basic linear algebra algorithms applied to Machine learning.

[P1673](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p1673r13.html) proposes a C++ Standard Library dense linear algebra interface based on the Basic Linear Algebra Subroutines (BLAS). This repo uses the reference implementation from [kokkos](https://github.com/kokkos) [stdBLAS](https://github.com/kokkos/stdBLAS) to create an implementation of the basic Neural Network backpropagation learning algorithm. The code is an evolved version of the code in the [Machine-Learning-CPP](https://github.com/GarethRichards/Machine-Learning-CPP) repo.

## Differences between [boost ublas](https://www.boost.org/doc/libs/1_81_0/libs/numeric/ublas/doc/index.html) and [std::linalg](https://en.cppreference.com/w/cpp/numeric/linalg)

Boost defines some basic container classes `ublas::vector` and `ublas::matrix`. These classes own the memory which the objects use. The corresponding class in the `std::linalg` world is `std::mdspan`. `mdspan` acts a view over a block of memory.

The types and constructors are detailed in the table below.
| boost ublas           | mdspan type                      | mdspan constructor   |
| -----------           | -------------------------------- | -------------------- |
| `ublas::vector(A)`    | `mdspan<T, dynamic_extent>`      | `(vec.data(), A)`    |
| `ublas::matrix(A, B)` | `mdspan<T, dextents<size_t, 2>>` | `(vec.data(), A, B)` |

And the basic usage is as follows.
``` cpp
    std::vector<float> x_vec(A);
    mdspan x(x_vec.data(), A);

    std::vector<float> mat_vec(A * B);
    mdspan mat(mat_vec.data(), A, B);   
```
`mdspan` has a number of other tricks - it controls the exact mapping of the array coordinates to the underlying memory. Thus, operations such as transpose just return a `mdspan` pointing at the same memory as the original one.
``` cpp
    std::vector<float> mat_vec(A * B);
    mdspan mat(x_vec.data(), A, B);  
    auto trans_mat = LinearAlgebra::transposed(mat);
    std::cout << mat.extent(0) << "," << mat.extent(1) << std::endl;
    std::cout << tras_mat.extent(0) << "," << tras_mat.extent(1) << std::endl;
``` 
In the above example `mat` and `trans_mat` point to the same piece of memory owner by `mat_vec`.

### Basic operations

The `boost::ublas` version of the Quadratic cost function is as follows.
``` cpp
T cost_fn(const ublas::vector<T> &a, const ublas::vector<T> &y) const {
    return 0.5 * pow(norm_2(a - y), 2);
}
```
The boost ublas vector has the +,-,*,/ operators the calculation `a - y` creates a temp vector which is consumed by the norm_2 function.

Using `std::linalg` our primitive is `mdspan` which does not have mathematical operators defined. At first glance `std::linalg` is lacking in only having an add function, but `mdspan` has some cool tricks to aid in this situation. `std::linalg` has two scaling functions `scale` transforms the data inside your `mdspan`, `scaled` on the other hand returns an `mdspan` which scales the data inside data when it's accessed. See the example below:

``` cpp
    std::vector<float> s = { 1, .4, 9 };
    mdspan<float, dextents<size_t, 2>> initial_span(s.data(), 3, 1);
    auto scaled_span = LinearAlgebra::scaled(2.0, initial_span);
    std::cout << "index,initial,scaled\n";
    for (int i = 0; i < initial_span.extent(0); ++i)
        std::cout << i << "," << initial_span(i, 0) << "," << scaled_span(i, 0) << "\n";
```

In the `std::linalg` version of the Quadratic cost function requires us to allocate some additional memory to save the intermediate results of the calculation. The boost ublas version of this function requires less code, the `std::linalg` version makes it clear which calculations are being used. 

``` cpp
using nvec = mdspan<T, dextents<size_t, 2>>;

static T cost_fn(const nvec& a, const nvec& y) {
    std::vector<T> y_data(y.extent(0));
    nvec yp(y_data.data(), y.extent(0), 1);
    LinearAlgebra::add(std::execution::par, a, LinearAlgebra::scaled(-1.0, y), yp);
    mdspan yp_v(y_data.data(), y.extent(0)); // convert to a 1D vector.
    return 0.5 * pow(LinearAlgebra::vector_norm2(std::execution::par, yp_v), 2);
    }
```
### Memory layout
`boost::ublas` stores its memory in vector and matrix classes. Internally the vector and matrix class store the data in a single memory buffer. The code to create the Biases and Weights data structures is shown here: 

``` cpp
    using BiasesVector = std::vector<ublas::vector<T>>;
    using WeightsVector = std::vector<ublas::matrix<T>>;

    BiasesVector biases;
    WeightsVector weights;

    for (auto i = 1; i < m_sizes.size(); ++i) {
        biases.emplace_back(ublas::zero_vector<T>{m_sizes[i]});
        weights.emplace_back(ublas::zero_matrix<T>{m_sizes[i], m_sizes[i - 1]});
    }
```

Using `mdspan` two objects are required, the `weights_data` and `biases_data` are just `std::vector` objects which store the underlying data and the `weights` and `biases` objects store a list of `mdspan` objects which map the underlying memory into our list of biases and weights. Instead of having a 1 to 1 mapping between `std::vector` objects and `mdspan` views, the code uses a single `std::vector`, `biases_data` to store the all the biases in the neural network. One technicality from this approach is you cannot rely on the generated `move` and `copy` copy constructors, so it is necessary to manually write them.

``` cpp
    using nvec = mdspan<T, dextents<size_t, 2>>;
    using nmatrix = mdspan<T, dextents<size_t, 2>>;
    using BiasesVector = std::vector<nvec>;
    using WeightsVector = std::vector<nmatrix>;

    std::vector<size_t> m_sizes;
    size_t max_vec_size = 0;
    size_t tot_vec_size = 0;
    BiasesVector biases;
    std::vector<T> biases_data;
    WeightsVector weights;
    std::vector<T> weights_data;

    size_t biases_data_size = 0;
    size_t weights_data_size = 0;
    for (auto i = 1; i < m_sizes.size(); ++i) {
        biases_data_size += m_sizes[i];
        weights_data_size += m_sizes[i] * m_sizes[i - 1];
    }
    biases_data.resize(biases_data_size, 0);
    weights_data.resize(weights_data_size, 0);
    biases_data_size = 0;
    weights_data_size = 0;
    for (auto i = 1; i < m_sizes.size(); ++i) {
        biases.push_back(nvec(&biases_data[biases_data_size], m_sizes[i], 1));
        weights.push_back(nmatrix(&weights_data[weights_data_size], m_sizes[i], m_sizes[i - 1]));
        biases_data_size += m_sizes[i];
        weights_data_size += m_sizes[i] * m_sizes[i - 1];
    }
```

### Feed Forward operation.
This operation returns the result of the input vector from a trained network.

![Neural net image](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

The Feed Forward operation requires one matrix multiplication of the input data with the trained weights matrix, the result is then added to the biases vector and finally the Neural activation function is called. The result is either input to the next layer of the neural net or the actual result. Thanks to the operator overloading in  `boost::ublas` it wins the less lines of code crown in this comparison.

``` cpp
    // Returns the output of the network if the input is a
    ublas::vector<T> feedforward(ublas::vector<T> a) const {
        for (auto i = 0; i < nd.biases.size(); ++i) {
            ublas::vector<T> c = prod(nd.weights[i], a) + nd.biases[i];
            this->Activation(c);
            a = c;
        }
        return a;
    }
```

In the `std::linalg` version of the calculation, we know how much memory we need in the intermediate layer at the start of the calculation. Thanks to the greater control of memory `mdspan` gives us, the number of memory allocations in this version is less than the `boost::ublas` version. Most of the extra code is due to manipulation of `mdspan` objects which are used in the calculation. 

``` cpp
    using nvec = mdspan<T, dextents<size_t, 2>>;

    void feedforward(nvec in_vec, nvec result) const override {
        std::vector<T> res_data(nd.tot_vec_size);
        size_t start_index = nd.m_sizes[0];
        size_t prev_index = 0;
        for (auto i = 0; i < nd.biases.size(); ++i) {
            // mdspan manipulation
            nvec resd(&res_data[start_index], nd.m_sizes[i + 1], 1);
            nvec& res = resd;
            if (i == nd.biases.size() - 1)
                res = result;
            nvec res_ind(&res_data[prev_index], nd.m_sizes[i], 1);
            nvec& res_in = res_ind;
            if (i == 0)
                res_in = in_vec;
            // Calculation
            LinearAlgebra::matrix_product(std::execution::par, nd.weights[i], res_in, res);
            LinearAlgebra::add(std::execution::par, res, nd.biases[i], res);
            this->Activation(res);
            prev_index = start_index;
            start_index += nd.m_sizes[i + 1];
        }
    }
```

# Final thoughts
I hope this brief introduction to `std::linalg` has shown you how to exciting new C++26 features. The `std::linalg` package allows you to perform the calculations on various calculation contexts and using `mspan` memory usage and optimisation is under the control of the programmer.

Personally I'd like to see many more examples of machine learning algorithms implemented using `std::linalg`.

# Further reading
## Neural Networks
If you want to learn more about Machine learning I can recommend the following resources
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen
[But what is a neural network? | Chapter 1, Deep learning](https://youtu.be/aircAruvnKk?si=50XAeNALzkgLZAps) by 3Blue1Brown
## BLAS
[stdBLAS](https://github.com/kokkos/stdBLAS)
[std::linalg](https://en.cppreference.com/w/cpp/numeric/linalg) from cpp reference
[boost:ublas](https://www.boost.org/doc/libs/1_84_0/libs/numeric/ublas/doc/)