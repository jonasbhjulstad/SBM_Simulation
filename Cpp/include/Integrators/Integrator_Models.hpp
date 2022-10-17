

#ifndef FROLS_MCMC_GENERATE_SIR_HPP
#define FROLS_MCMC_GENERATE_SIR_HPP

#include <array>
#define SUNDIALS_SINGLE_PRECISION

#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>
#include <fstream>
#include <iostream>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <vector>
#include <array>

namespace FROLS::Integrators {
    template<uint16_t Nx, uint16_t Nt, typename Derived, typename Param>
    struct Model_Integrator {
        Model_Integrator(float t0 = 0.): t_current(t0){}
        typedef std::array<float, Nx> State;
        typedef std::array<std::array<float, Nt + 1>, Nx> Trajectory;
        float t_current;

        State step(const State &x) {
            return static_cast<Derived *>(this)->step(x);
        }

        std::pair<Trajectory, std::array<float, Nt+1>> simulate(const std::array<float, 3>& x0) {
            std::array<std::array<float, Nt + 1>, Nx> trajectory;
            std::array<float, Nt+1> t;
            t[0] = t_current;
            auto x = x0;
            for (int j = 0; j < Nx; j++)
            {
                trajectory[j][0] = x0[j];
            }

            for (int i = 0; i < Nt; i++) {
                x = step(x);
                t[i+1] = t_current;
                for (int j = 0; j < Nx; j++)
                {
                    trajectory[j][i+1] = x[j];
                }
            }
            return std::make_pair(trajectory, t);
        }
    };

    template<uint16_t Nx, uint16_t Nt, class Derived, typename Param>
    struct CVODE_Integrator : public Model_Integrator<Nx, Nt, CVODE_Integrator<Nx, Nt, Derived, Param>, Param> {
        using Base = Model_Integrator<Nx, Nt, CVODE_Integrator<Nx, Nt, Derived, Param>, Param>;
        using State = typename Base::State;
        using Trajectory = typename Base::Trajectory;
        SUNContext ctx;
        void *cvode_mem;
        float dt = 0;
        float abs_tol;
        float rel_tol;
        float& t_current = Base::t_current;
        SUNMatrix jac_mat;
        N_Vector x;
        SUNLinearSolver solver;
        CVODE_Integrator(const std::array<float, Nx>& x0,float dt,
                         float abs_tol = 1e-5, float reltol = 1e-5,
                         float t0 = 0)
                : dt(dt), abs_tol(abs_tol), rel_tol(reltol),
                  Model_Integrator<Nx, Nt, CVODE_Integrator<Nx, Nt, Derived, Param>, Param>(t0){

            SUNContext_Create(NULL, &ctx);
            if (check_flag((void *) ctx, "SUNContextCreate", 0))
                return;
            x = N_VNew_Serial(Nx, ctx);
            for (int i = 0; i < Nx; i++)
            {
                NV_Ith_S(x, i) = x0[i];
            }
            jac_mat = SUNMatNewEmpty(ctx);
            cvode_mem = CVodeCreate(CV_BDF, ctx);
        }
        State step(const std::array<float, Nx> x_current) {
//            N_Vector x = N_VNew_Serial(Nx, ctx);
            for (int i = 0; i < Nx; i++)
            {
                NV_Ith_S(x, i) = x_current[i];
            }
            int flag = CVode(cvode_mem, t_current + dt, x, &t_current, CV_NORMAL);
            // copy x to Vec
            std::array<float, Nx> x_next;
            for (int i = 0; i < Nx; i++) {
                x_next[i] = NV_Ith_S(x, i);
                if (check_flag(&flag, "CVode", 1))
                    break;
            }
            return x_next;
        }

        ~CVODE_Integrator() {
            N_VDestroy(x);
            CVodeFree(&cvode_mem);
            SUNLinSolFree(solver);
            SUNContext_Free(&ctx);
        }

        int initialize_solver(CVRhsFn f, CVLsJacTimesVecFn jac, void *user_data) {
            int flag;
            solver = SUNLinSol_SPGMR(x, PREC_NONE, 0, ctx);
            flag = CVodeInit(cvode_mem, f, t_current, x);

            flag = CVodeSetUserData(cvode_mem, user_data);
            if (check_flag(&flag, "CVodeSetUserData", 1))
                return EXIT_FAILURE;

            if (check_flag(&flag, "CVodeSetUserData", 1))
                return EXIT_FAILURE;

            flag = CVodeSStolerances(cvode_mem, rel_tol, abs_tol);
            if (check_flag(&flag, "CVodeSStolerances", 1))
                return EXIT_FAILURE;

            // Set linear solver as iterative
            flag = CVodeSetLinearSolver(cvode_mem, solver, NULL);
            if (check_flag(&flag, "CVodeSetLinearSolver", 1))
                return EXIT_FAILURE;

            // Assign jacobian function
            flag = CVodeSetJacTimes(cvode_mem, NULL, jac);
            if (check_flag(&flag, "CVodeSetJacTimes", 1))
                return EXIT_FAILURE;
            return EXIT_SUCCESS;
        }

    private:
        static int check_flag(void *flagvalue, const char *funcname, int opt) {
            int *errflag;

            /* Check if SUNDIALS function returned NULL pointer - no memory allocated
             */
            if (opt == 0 && flagvalue == NULL) {
                fprintf(stderr,
                        "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                        funcname);
                return EXIT_FAILURE;
            }

                /* Check if flag < 0 */
            else if (opt == 1) {
                errflag = (int *) flagvalue;
                if (*errflag < 0) {
                    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                            funcname, *errflag);
                    return EXIT_FAILURE;
                }
            }

                /* Check if function returned NULL pointer - no memory allocated */
            else if (opt == 2 && flagvalue == NULL) {
                fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                        funcname);
                return EXIT_FAILURE;
            }

            return EXIT_SUCCESS;
        }
    };
} // namespace FROLS::Integrators
#endif
