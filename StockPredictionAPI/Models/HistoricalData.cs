namespace StockPredictionAPI.Models
{
    public class HistoricalData
    {
        public string Date { get; set; }
        public float Open { get; set; }
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public float Volume { get; set; }
        public float EMA { get; set; }
    }
}