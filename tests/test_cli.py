# tests/test_cli.py
import logging
import unittest
from unittest.mock import patch, mock_open
import sys
from pyvizualizer.cli import main


class TestCLI(unittest.TestCase):

    @patch('pyvizualizer.cli.open', new_callable=mock_open)
    @patch('pyvizualizer.cli.setup_logging')
    @patch('pyvizualizer.cli.Analyzer')
    @patch('pyvizualizer.cli.MermaidGenerator')
    @patch('sys.argv', ['cli.py', 'test_project', '--output', 'test_output.mmd', '--log-level', 'DEBUG'])
    def test_main(self, mock_analyzer, mock_mermaid_generator, mock_setup_logging, mock_open):
        mock_analyzer_instance = mock_analyzer.return_value
        mock_analyzer_instance.analyze.return_value = 'test_project'

        mock_mermaid_generator_instance = mock_mermaid_generator.return_value
        mock_mermaid_generator_instance.generate.return_value = 'mock_mermaid_code'

        main()

        mock_setup_logging.assert_called_once_with(logging.DEBUG)
        mock_analyzer.assert_called_once_with('test_project')
        mock_analyzer_instance.analyze.assert_called_once()
        mock_mermaid_generator.assert_called_once_with('test_project')
        mock_mermaid_generator_instance.generate.assert_called_once()
        mock_open.assert_called_once_with('test_output.mmd', 'w', encoding='utf-8')
        mock_open().write.assert_called_once_with('mock_mermaid_code')


if __name__ == '__main__':
    unittest.main()