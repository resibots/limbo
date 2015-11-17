Limbo Framework and Basic Example
=================================================

Limbo Framework
----------------------------

Basic Example
----------------------------

Create directories and files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say we want to create an experiment called "test". The first thing to do is to create the folder ``exp/test`` under the limbo root. Then add two files:

* the ``main.cpp`` file
* a pyhton file called ``wscript``, which will be used by WAF to register the executable for building

The ``wscript`` file has to have the following content: ::

    def options(opt):
        pass


    def build(bld):
        bld(features='cxx cxxprogram',
            source='main.cpp',
            includes='. ../../src',
            target='test',
            uselib='BOOST EIGEN TBB',
            use='limbo') 

For this example, we will optimize a simple function: :math:`-{(5*x - 2.5)}^2 + 5`.
To do it, we have to define the evaluation function for the optimizer to call: ::

    struct eval {
        static constexpr size_t dim_in = 1;
        static constexpr size_t dim_out = 1;

        Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
        {
            Eigen::VectorXd res(1);
            res(0) = -((5 * x(0) - 2.5) * (5 * x(0) - 2.5)) + 5;
            return res;
        }
    }

It is required that the evaluation struct has 