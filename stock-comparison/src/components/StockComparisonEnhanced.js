import React, { useState, useEffect, cloneElement } from 'react';
import axios from 'axios';
import StockChart from './StockChart';
import { TooltipProvider } from './TooltipContext';
import 'chart.js/auto';
import '../Styles/StockComparison.css';

const apiKey = 'fdBhj6FJhbAKaYyTh5fU3pwUvPY5X32E';

const fetchStock = async (symbol) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}?apikey=${apiKey}`);
    const data = response.data.historical;
    return data.map(entry => ({
        date: entry.date,
        open: entry.open,
        high: entry.high,
        low: entry.low,
        close: entry.close,
        volume: entry.volume
    }));
};

const fetchIndicatorData = async (symbol, indicator) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/technical_indicator/1day/${symbol}?type=${indicator}&period=1825&apikey=${apiKey}`);
    const data = response.data;
    return data.map(entry => ({
        x: entry.date,
        y: entry[indicator]
    }));
};

const fetchHistoricalRating = async (symbol) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/historical-rating/${symbol}?apikey=${apiKey}`);
    const data = response.data;
    return data.map(entry => ({
        x: entry.date,
        y: entry.ratingScore,
        recommendation: entry.ratingRecommendation
    }));
};

const fetchLSTMPrediction = async (symbol) => {
    const response = await axios.get(`http://localhost:5143/api/LSTMStockPrediction/${symbol}`);
    return response.data;
};

const fetchMLPrediction = async (symbol) => {
    const response = await axios.get(`http://localhost:5143/api/StockPrediction/${symbol}`);
    return response.data;
};

const fetchStockDetails = async (symbol) => {
        const response = await axios.get(`https://financialmodelingprep.com/api/v3/quote/${symbol}?apikey=${apiKey}`);
        return response.data[0];
};

const fetchNewsSentiment = async (symbol) => {
    const response = await axios.get(`https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=${symbol}&apikey=${apiKey}`);
    return response.data;
};

const StockComparisonEnhanced = () => {
    const [symbol, setSymbol] = useState('');
    const [stockDetails, setStockDetails] = useState(null);
    const [priceData, setPriceData] = useState([]);
    const [emaData, setEmaData] = useState([]);
    const [rsiData, setRsiData] = useState([]);
    const [ratingData, setRatingData] = useState([]);
    const [lstmpredictionData, setLSTMPredictionData] = useState(null);
    const [mlpredictionData, setMLPredictionData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [recommendation, setRecommendation] = useState("");
    const [newsSentiment, setNewsSentiment] = useState(null);
    const [recommendationFontColor, setRecommendationFontColor] = useState("black");

    const fetchData = async (symbol) => {
        if (!symbol) {
            alert('Please enter a stock symbol');
            return;
        }

        setPriceData([]);
        setEmaData([]);
        setStockDetails(null);
        setRsiData([]);
        setRatingData([]);
        setLSTMPredictionData(null);
        setMLPredictionData(null);
        setRecommendation("");
        setNewsSentiment(null);
        setRecommendationFontColor("black");
        setLoading(true);
        try {
            const priceData = await fetchStock(symbol);
            const lstmpredictionData = await fetchLSTMPrediction(symbol);
            const mlpredictionData = await fetchMLPrediction(symbol);
            let emaData = await fetchIndicatorData(symbol, "ema");
            let rsiData = await fetchIndicatorData(symbol, "rsi");
            let ratingData = await fetchHistoricalRating(symbol);
            let stockDetails = await fetchStockDetails(symbol);
            let newsSentiment = await fetchNewsSentiment(symbol);
            let recommendation = ratingData[0]?.recommendation;
            let recommendationFontColor = "black";
            if(recommendation === "Buy"){
                recommendationFontColor = "blue";
            }
            else if(recommendation === "Neutral"){
                recommendation = "Hold";
                recommendationFontColor = "green";
            }
            else if(recommendation === "Sell"){
                recommendationFontColor = "red";
            }
            // Calculate the date 1825 days before today
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - 1825);

            // Filter the data
            emaData = emaData.filter(entry => new Date(entry.x) >= cutoffDate).sort((a, b) => new Date(a.x) - new Date(b.x));;
            rsiData = rsiData.filter(entry => new Date(entry.x) >= cutoffDate).sort((a, b) => new Date(a.x) - new Date(b.x));;
            ratingData = ratingData.filter(entry => new Date(entry.x) >= cutoffDate).sort((a, b) => new Date(a.x) - new Date(b.x));;
            setPriceData(priceData);
            setEmaData(emaData);
            setRsiData(rsiData);
            setRatingData(ratingData);
            setStockDetails(stockDetails);
            setRecommendation(recommendation);
            setNewsSentiment(newsSentiment);
            setRecommendationFontColor(recommendationFontColor);
            setLSTMPredictionData(lstmpredictionData);
            setMLPredictionData(mlpredictionData);
            console.log('LSTM Prediction:', lstmpredictionData);
            console.log('ML Prediction:', mlpredictionData);
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <TooltipProvider>
            <div className="container">
            <div className="main-content">
                <div className="side-content">
                    <div>
                        {stockDetails && (
                            <div>
                                <p style ={{fontSize:"20px"}}>
                                <b>{stockDetails.name}
                                <span>({stockDetails.symbol})</span></b>
                                </p>
                                <span style ={{fontSize:"48px"}}><b>{stockDetails.price} </b></span>
                                <span>{stockDetails.changesPercentage >= 0 ? <span style={{color:"green"}} >{'\u25B2'}</span> : <span style={{color:"red"}}>{'\u25BC'}</span>}</span>
                                <span style={{color:"green"}}><b>{stockDetails.changesPercentage}% </b></span>
                                <span style={{color:"green"}}>({stockDetails.change})</span>
                            </div>
                        )}
                    </div>
                </div>
                <div>
                <h1 style={{textAlign:"center"}}>Stock Forecasting</h1>
                    <div style={{marginLeft: "180px", justifyContent:'center'}}>
                        <input
                            type="text"
                            id="stockSymbolInput"
                            placeholder="Enter Stock Symbol"
                        />
                        <button style={{marginLeft : "10px"}} onClick={() => {
                                const symbol = document.getElementById('stockSymbolInput').value;
                                fetchData(symbol);
                            }}>Submit</button>
                        <div>
                            Stock Recommendation : 
                            {loading ? (
                                <span className="loading ml-5">Loading...</span>
                            ) : ( <span style={{fontColor : recommendationFontColor}}><b>{recommendation}</b></span>)}
                        </div>
                        {newsSentiment && (
                            <div>
                                <p>News Sentiment: <b>{newsSentiment.feed[0].overall_sentiment_label}({newsSentiment.feed[0].overall_sentiment_score}</b>)</p>
                            </div>
                        )}
                    </div>
                </div>
                <div className="side-content">
                    { lstmpredictionData && (
                    <div className="metrics-container">
                        <h3>LSTM Model Performance Metrics</h3>
                        <div>Train data : {lstmpredictionData?.trainDataRange.startDate} to {lstmpredictionData?.trainDataRange.endDate}</div>
                        <div>Test data : {lstmpredictionData?.testDataRange.startDate} to {lstmpredictionData?.testDataRange.endDate}</div>
                        <div>Mean Absolute Error (MAE): {lstmpredictionData?.metrics.meanAbsoluteError.toFixed(2)}</div>
                        <div>Root Mean Squared Error (RMSE): {lstmpredictionData?.metrics.rootMeanSquaredError.toFixed(2)}</div>
                    </div>
                    )}
                </div>
            </div>
                <>
                    <div className="main-content">
                        <div className="part-content">
                            <div className="table-container scrollable-table">
                                <h4>Stock Data Table</h4>
                                {priceData.length > 0 ? (
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Date</th>
                                                <th>Open</th>
                                                <th>High</th>
                                                <th>Low</th>
                                                <th>Close</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {priceData.sort((a, b) => new Date(b.date) - new Date(a.date)).map((entry, index) => (
                                                <tr key={index}>
                                                    <td>{new Date(entry.date).toLocaleDateString()}</td>
                                                    <td>{entry.open}</td>
                                                    <td>{entry.high}</td>
                                                    <td>{entry.low}</td>
                                                    <td>{entry.close}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                ) : (
                                    <p>No data available</p>
                                )}
                            </div>
                        </div>
                        <div className='side-content'>
                            <div className="chart-container">
                                <StockChart title = "Stock Price comparison" canvasId="priceChart" datasets={[
                                    {
                                        label: 'Historical Price',
                                        data: priceData.map(entry => ({ x: entry.date, y: entry.close })),
                                        borderWidth: 1,
                                        borderColor: 'rgba(75, 192, 192, 1)',
                                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                        fill: true,
                                    },
                                    {
                                        label: 'LSTM Predicted Close',
                                        data: lstmpredictionData?.forecastedAdjClose.map((entry) => ({
                                            x: entry.date,
                                            y: entry.forecastedClose
                                        })) || [],
                                        borderWidth: 1,
                                        borderColor: 'rgba(255, 99, 132, 1)',
                                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                        fill: true,
                                    },
                                    {
                                        label: 'Microsoft ML TimeSeries Predicted Close',
                                        data: mlpredictionData?.forecast.map((entry) => ({
                                            x: entry.date,
                                            y: entry.forecastedClose
                                        })) || [],
                                        borderWidth: 1,
                                        borderColor: 'rgba(54, 162, 235, 1)',
                                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                        fill: true,
                                    }
                                ]} className="chart" />
                            </div>
                        </div>
                        <div className="part-content">
                            <div className="table-container scrollable-table">
                                <h4>Stock Prediction Table</h4>
                                {lstmpredictionData && mlpredictionData ? (
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Date</th>
                                                <th>Actual Close</th>
                                                <th>ML Predicted Close</th>
                                                <th>LSTM Predicted Close</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {lstmpredictionData?.forecastedAdjClose.map((d, index) => {
                                                const date = new Date(d.date).toLocaleDateString();
                                                const actualClose = priceData.find(entry => new Date(entry.date).toLocaleDateString() === date)?.close || 'N/A';
                                                return (
                                                    <tr key={index}>
                                                        <td>{date}</td>
                                                        <td>{actualClose}</td>
                                                        <td>{mlpredictionData.forecast.find(entry => new Date(entry.date).toLocaleDateString() === date)?.forecastedClose || 'N/A'}</td>
                                                        <td>{d.forecastedClose}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                ) : (
                                    <p>No data available</p>
                                )}
                            </div>
                        </div>
                    </div>
                    <div className="chart-wrapper">
                        <div className="chart-container">
                            <StockChart title = "EMA" canvasId="emaChart" datasets={[{
                                label: 'EMA',
                                data: emaData,
                                borderWidth:1,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                fill: true,
                            }]} className="chart" />
                        </div>
                        <div className="chart-container">
                            <StockChart title="RSI" canvasId="rsiChart" datasets={[{
                                label: 'RSI',
                                data: rsiData,
                                borderWidth:1,
                                borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                fill: true,
                            }]} className="chart" />
                        </div>
                        <div className="chart-container">
                            <StockChart title="Historical Rating" canvasId="ratingChart" datasets={[{
                                label: 'Rating Score',
                                data: ratingData,
                                borderWidth:1,
                                borderColor: ' rgba(255, 159, 64, 1)',
                                backgroundColor: ' rgba(255, 159, 64, 0.2)',
                                fill: true,
                            }]} className="chart" />
                        </div>
                    </div>
                </>
            </div>
        </TooltipProvider>
    );
};

export default StockComparisonEnhanced;