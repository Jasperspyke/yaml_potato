import random as rn
import Support


def main():
    fate = rn.choices(0,3)
    print(fate)
    if not fate:
        print('You cannot rest when there are monsters nearby')
    elif fate == 1:
        print('Steady state')
    elif fate == 2:
        print('take your time ')
    elif fate == 3:
        print('whatever it takes')
    elif fate == 4:
        print('go hard and go home')


if __name__ == '__main__':
    main()