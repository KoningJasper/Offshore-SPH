import argparse, sys
import examples.dam_break_complete

def test(args):
    raise print('Testing hasn\'t been implemented yet.')

def run(args):
    if args.file.lower() == 'dambreak':
        examples.dam_break_complete.main()
    else:
        print('Invalid file to run.')

def main():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Runner', description='The main entry point for the Offshore-SPH project. Using this runner example tests can be run or the test framework can be run, to verify the integrity of the project.')
    subparsers = parser.add_subparsers()

    # Test
    parser_test = subparsers.add_parser('test', help='Run tests for the project.')
    parser_test.add_argument('file', type=str, help='The test to run.')
    parser_test.set_defaults(func=test)

    # Run
    parser_run = subparsers.add_parser('run', help='Run one of the many examples.')
    parser_run.add_argument('file', type=str, help='The file to run, currently the only file that can be run is the dambreak.')
    parser_run.set_defaults(func=run)

    # Execute the argument
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()