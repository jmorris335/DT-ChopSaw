import concurrent.futures

# from src.test.scratchpaper import scratchpaper
# from src.sim.dev_sim import dev_sim_main
from src.sim.mocap_sim import mocapSim
from src.gui.app_test import run as dash_app
# from src.gui.skeleton import skeletonTest

def main():
    # mocapSim()
    dash_mocap_sim()
    # dev_sim_main()
    # skeletonTest()

def dash_mocap_sim():
    # dash_app()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(dash_app)
        executor.submit(mocapSim)

if __name__ == '__main__':
    main()