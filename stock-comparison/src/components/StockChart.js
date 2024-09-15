// src/components/StockChart.js
import React from 'react';
import { Chart, registerables } from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import 'chartjs-adapter-date-fns';

Chart.register(...registerables, zoomPlugin);

class StockChart extends React.Component {
    chartInstance = null;

    componentDidMount() {
        this.createChart();
    }

    componentDidUpdate() {
        this.createChart();
    }

    componentWillUnmount() {
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }
    }

    createChart() {
        const { canvasId, datasets } = this.props;
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }

        this.chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: datasets
            },
            options: {
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            tooltipFormat: 'PP', // Display full date in tooltip
                            displayFormats: {
                                day: 'MMM d, yyyy' // Display date and year on x-axis
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y;
                                }
                                return label;
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                        }
                    }
                },
                hover: {
                    mode: 'index',
                    intersect: false
                }
            }
        });
    }

    render() {
        return <canvas id={this.props.canvasId}></canvas>;
    }
}

export default StockChart;