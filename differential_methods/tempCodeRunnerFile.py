    for sys in systems:
        solver = HeunSolver(sys)
        n_steps = int((sys.x_end - sys.x0) / sys.h)
        solver.solve(n_steps)
        results_all.append((sys.label, solver.solution))
        print(f"\n==> System: {sys.label}")
        solver.print_results()