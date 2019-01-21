import argparse
import examples.dam_break_complete


# Commands
def test(file: str):
    raise Exception('Testing hasn\'t been implemented yet.')

def run(file: str):
    if file.lower() == 'dambreak':
        examples.dam_break_complete.main()
    else:
        raise Exception('Invalid file to run.')

def main():
    # create the top-level parser
    parser = argparse.ArgumentParser(description='The main entry point for the Offshore-SPH project. Using this runner example tests can be run or the test framework can be run, to verify the integrity of the project.')
    subparsers = parser.add_subparsers()

    test = subparsers.add_parser('test', help='Run tests for the project.')
    test.add_argument('file', type=str, help='The test to run.', required=True)
    run = subparsers.add_parser('run', help='Run one of the many examples.')
    run.add_argument('file', type=str, help='The file to run, currently the only file that can be run is the dambreak.', required=True)

    args = parser.parse_args()
    print(args.accumulate(args.run))


if __name__ == '__main__':
    main()