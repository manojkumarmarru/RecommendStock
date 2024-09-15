namespace StockPredictionAPI.Models
{
    public class HistoricalDataResponse
    {
        public string Symbol { get; set; }
        public List<HistoricalData> Historical { get; set; }
    }
}