# test_analyzer.py

import unittest
from unittest.mock import patch, MagicMock
from pyvizualizer import CodeGraph, Analyzer


class TestAnalyzer(unittest.TestCase):

    @patch('pyvizualizer.analyzer.get_python_files')
    @patch('pyvizualizer.analyzer.CodeParser')
    @patch('pyvizualizer.analyzer.CodeGraph')
    def test_analyze(self, MockCodeGraph, MockCodeParser, mock_get_python_files):
        # Setup mock return values
        mock_get_python_files.return_value = ['file1.py', 'file2.py']
        mock_parser_instance = MockCodeParser.return_value
        mock_graph_instance = MockCodeGraph.return_value

        # Create an instance of Analyzer
        analyzer = Analyzer('dummy_project_path')

        # Call the analyze method
        result = analyzer.analyze()

        # Assertions
        mock_get_python_files.assert_called_once_with('dummy_project_path')
        mock_parser_instance.parse_files.assert_called_once_with(['file1.py', 'file2.py'])
        mock_graph_instance.build_graph.assert_called_once_with(mock_parser_instance.definitions)
        self.assertEqual(result, mock_graph_instance)


class TestAnalyzerWithSampleProject(unittest.TestCase):

    def test_analyze_sample_project(self):
        # Path to the sample project
        sample_project_path = 'C:/Users/SRIZVI13/PycharmProjects/PyVizualizer/examples/sample_project'

        # Create an instance of Analyzer
        analyzer = Analyzer(sample_project_path)

        # Call the analyze method
        result = analyzer.analyze()

        for node in result.nodes:
            print(node)

        # Retrieve edges using the get_edges method
        edges = result.get_edges()

        # Assertions
        self.assertIsInstance(result, CodeGraph)
        self.assertTrue(result.nodes)  # Ensure the graph has nodes
        #self.assertTrue(edges)  # Ensure the graph has edges

if __name__ == '__main__':
    unittest.main()