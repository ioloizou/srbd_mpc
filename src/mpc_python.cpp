#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "mpc.hpp"

namespace py = pybind11;

PYBIND11_MODULE(g1_mpc_py, m) {
    m.doc() = "Python bindings for G1 MPC controller";

    py::class_<g1_mpc::MPC>(m, "MPC")
        .def(py::init<double, double, double, double, int>(),
             py::arg("mu") = 0.3,
             py::arg("fz_min") = 10.0,
             py::arg("fz_max") = 666.0,
             py::arg("dt") = 0.04,
             py::arg("horizon_length") = 15)
        .def_property("x0", &g1_mpc::MPC::getX0, &g1_mpc::MPC::setX0)
        .def_property("x_ref_hor", &g1_mpc::MPC::getXRefHor, &g1_mpc::MPC::setXRefHor)
        .def_property("u_opt", &g1_mpc::MPC::getUOpt, &g1_mpc::MPC::setUOpt)
        .def_property("x_opt", &g1_mpc::MPC::getXOpt, &g1_mpc::MPC::setXOpt)
        .def_property("u_opt0", &g1_mpc::MPC::getUOpt0, &g1_mpc::MPC::setUOpt0)
        .def_property("r", &g1_mpc::MPC::getr, &g1_mpc::MPC::setr)
        .def_property("p_com_horizon", &g1_mpc::MPC::getPComHorizon, &g1_mpc::MPC::setPComHorizon)
        .def_property("c_horizon", &g1_mpc::MPC::getCHorizon, &g1_mpc::MPC::setCHorizon)
        .def_readonly_static("HORIZON_LENGTH", &g1_mpc::MPC::horizon_length_)
        .def_readonly_static("g", &g1_mpc::MPC::g_)
        .def("init_matrices", &g1_mpc::MPC::initMatrices)
        .def("extract_psi", &g1_mpc::MPC::extractPsi)
        .def("calculate_rotation_matrix_T", &g1_mpc::MPC::calculateRotationMatrixT)
        .def("set_Q", &g1_mpc::MPC::setQ)
        .def("set_R", &g1_mpc::MPC::setR)
        .def("calculate_A_continuous", &g1_mpc::MPC::calculateAContinuous)
        .def("calculate_A_discrete", &g1_mpc::MPC::calculateADiscrete)
        .def("calculate_B_continuous", &g1_mpc::MPC::calculateBContinuous)
        .def("calculate_B_discrete", &g1_mpc::MPC::calculateBDiscrete)
        .def("calculate_Aqp", &g1_mpc::MPC::calculateAqp)
        .def("calculate_Bqp", &g1_mpc::MPC::calculateBqp)
        .def("calculate_Ac", &g1_mpc::MPC::calculateAc)
        .def("calculate_bounds", &g1_mpc::MPC::calculateBounds)
        .def("calculate_hessian", &g1_mpc::MPC::calculateHessian)
        .def("calculate_gradient", &g1_mpc::MPC::calculateGradient)
        .def("solve_qp", &g1_mpc::MPC::solveQP)
        .def("compute_rollout", &g1_mpc::MPC::computeRollout,
             py::arg("x_current") = nullptr,
             py::arg("only_first_step") = false)
        .def("update", &g1_mpc::MPC::update,
             py::arg("contact"),
             py::arg("c_horizon"),
             py::arg("p_com_horizon"),
             py::arg("x_current") = nullptr,
             py::arg("one_rollout") = false,
             py::return_value_policy::reference_internal);
}
