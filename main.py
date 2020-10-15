import gym
import simple_driving


def main():
    env = gym.make("SimpleDriving-v0")
    env.seed(1)


if __name__ == '__main__':
    main()
