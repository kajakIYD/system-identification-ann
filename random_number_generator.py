from random import gauss
from random import uniform
from random import seed


def generate_white_noise_probe():
    return gauss(0.0, 1.0)


def generate_uniform_noise_probe(a=-1, b=1):
    return uniform(a, b)


def main():
    # seed random number generator

    # create white noise series
    print(generate_uniform_noise_probe())
    print(generate_uniform_noise_probe())


if __name__ == "__main__":
    main()