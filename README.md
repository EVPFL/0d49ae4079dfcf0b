# EVPFL 0.0.1

This repository contains the codes for the paper
> [EVPFL](XXX)

Our code is based on the codes for 
[fedavgpy](https://github.com/bokunwang/fedavgpy), 
[HEAAN-Python](https://github.com/Huelse/HEAAN-Python) is a Python wrapper for HEAAN library.
[MKLHS] The Multi-key Linearly Homomorphic Signature Scheme is on the [paper](https://eprint.iacr.org/2019/830.pdf), which is implemented in the library [relic](https://github.com/relic-toolkit/relic).
[LHH] Linearly Homomorphic Hash supports linear operations on its output. This scheme is based on the LHH funtion used in [paper](https://eprint.iacr.org/2022/1073) and the library [VeriFL](https://github.com/ErwinSCat/VeriFL)

## DIR TREE
``` shell
|-README.md
|-main.py
|-Datasets (based on [fedavgpy])
    |-cifar10
        |-generate_equal.py
    |-mnist
        |-generate_equal.py
|-fedAvg (based on [fedavgpy])
|-mkTPFL
    |-ckks (Classes based on MPHEAAN)
        |-ckks_decryptor.py
        |-ckks_encoder.py
        |-ckks_encryptor.py
        |-ckks_evaluator.py
        |-ckks_key_generator.py
        |-ckks_parameters.py
        |-util.py
    |-mklhs (Classes based on MKLHE)
        |-mklhs_encoder.py
        |-mklhs_parameters.py
        |-mklhs_verifier.py
    |-roles
        |-dataisland.py (Client Class)
        |-flserver.py (Server Class)
        |-model_evaluator.py (Encrypted Model Evaluator Class: to evaluate the ciphertext of local updates)
    |-srcs-cpython
        |-LHH-Python (Python wrapper for the Linearly Homomorphic Hash library)
        |-MKLHS-Python (Python wrapper for the Multi-key Linearly Homomorphic Signature library)
        |-MPHEAAN-Python (Python wrapper for the MP-HEAAN library)
        |-pybind11
    |-trainers
        |-base.py (basic Trainer Class)
        |-fedavg.py (fedavg Trainer Class and correspoding train Function)
        |-sa.py (SA Trainer Class: for verficable aggregation test)
        |-se.py (SE Trainer Class: for secure evaluation test)
    |-utils
|-pfl_setup_datas (public paramaters and datas generated in setup phase)
    |-dikeys (keys of each client)
    |-svkeys (keys of server)
    |-diskshares (sk shares holded by each clients)
|-result (logs)
|-tests
    |-test_va_cifar10.py (verificable aggregation on CIFAR10 Dataset)
    |-test_va_mnist.py (verificable aggregation on MNIST Dataset)
    |-test_se.py (basic secure evaluation test: a local update contains local gradients of LeNet5 training on a local MNIST dataset)
    |-tests_se.py (execute test_va.py multiple times: different evaluation methods and models training on different datasets)
    |-test_va.py (basic verificable aggregation test: a local update is a random verctor)
    |-tests_va.py (execute test_va.py multiple times: different lengths and numbers of aggregated inputs)

```



## Usage

### 1. Compile libraries: LHH,MKLHS,MPHEAAN

    Please read their README.md in corresponding directories firstly.

  * ### pybind11 (mkTPFL/srcs-cpython/pybind11)
    ``` shell
        cd mkTPFL/srcs-cpython/pybind11
        mkdir build && cd build
        cmake ..
        make
        sudo make install
    ```


  * ### MPHEAAN (mkTPFL/srcs-cpython/MPHEAAN-Python)
    (the compiled python library is named HEAAN)

    install GMP and NTL first
    Download links:
        GMP: https://gmplib.org/#DOWNLOAD
        NTL: https://libntl.org/download.html

    ``` shell
        # GMP-6.2.0
        sudo apt-get install m4
        cd gmp-6.2.0
        ./configure SHARED=on
        make
        make check
        sudo make install

        # NTL-11.4.3
        cd ntl-11.4.3/src
        ./configure SHERED=on
        make
        make check
        sudo make install
    ```
    #### Q&A
        problem: install ntl error: gmp version mismatch 
        solve: cd gmp; sudo make uninstall; and install right version (gmp-6.2.0+NTL-11.4.3) 

    ```shell
        cd mkTPFL/srcs-cpython/MKLHS-Python

        # MP-HEAAN
        cd MP-HEAAN/lib
        make all

        # MP-HEAAN-Python (global install)  
        python setup.py build_ext -i
        sudo python setup.py install 

        # test
        python tests/test-basic.py
    ```


  * ### LHH (mkTPFL/srcs-cpython/LHH-Python)
    install openssl first
    Mac Download:
        ``` shell
            brew install openssl@1.1
        ```

    ``` shell
        cd mkTPFL/srcs-cpython/LHH-Python

        # LHH (build "libLHH.a")
        cd LHH/lib
        make all 

        # LHH-Python (global install)
        cd LHH-Python/
        sudo python setup.py install

        # test
        python test.py
    ```
    #### Q&A
        problem: install ntl error: gmp version mismatch 
        solve: cd gmp; sudo make uninstall; and install right version (gmp-6.2.0+NTL-11.4.3) 


  * ### MKLHS (mkTPFL/srcs-cpython/MKLHS-Python)
    ``` shell
        cd mkTPFL/srcs-cpython/MKLHS-Python

        # relic (build "./target/")
        cd MKLHS/relic
        unzip src.zip 
        make
    ```
    #### Q&A
        Problem: errors in building ./target
        solve: modify the MKLHS/relic/Makefile 
        change "/preset/arm64-pbc-bls12-381.sh" to other /preset/xx.sh

    ``` shell
        # MKLHS (build "libMKLHS.a")
        cd MKLHS/lib
        make all
    ```
    #### Q&A
        Problem: fatal error: 'relic.h' file not found
        solve: sudo make all 

     ``` shell
        # MKLHS-Python (global install)
        cd MKLHS-Python
        sudo python setup.py install 

        # test
        python test.py
    ```


###  2. Generate datasets. 
* ### CIFAR10
    ``` shell
        cd Datasets/cifar10
        python generate_equal.py
    ```
* ### MNIST
    ``` shell
        cd Datasets/mnist
        python generate_equal.py
    ```


### 3. Then start to train. You can run the tests or main.py.
``` shell
    python main.py
```

``` shell
    python tests/test_va.py
```

