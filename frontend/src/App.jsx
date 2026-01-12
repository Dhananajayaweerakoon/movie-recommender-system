import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [movieName, setMovieName] = useState('')
  const [recommendations, setRecommendations] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const getRecommendations = async () => {
    if (!movieName) return
    
    setLoading(true)
    setError('')
    setRecommendations([])

    try {
      // Connect to your FastAPI Backend
      const response = await axios.get(`http://127.0.0.1:8000/recommend/${movieName}`)
      setRecommendations(response.data.recommendations)
    } catch (err) {
      setError("Movie not found! Try another name.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <h1>🎬 Movie Recommender AI</h1>
        <p>Type a movie you love, and we'll suggest 10 more.</p>
      </header>

      <div className="search-box">
        <input 
          type="text" 
          placeholder="e.g. Avatar, Batman, Inception..." 
          value={movieName}
          onChange={(e) => setMovieName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && getRecommendations()}
        />
        <button onClick={getRecommendations}>
          Recommend
        </button>
      </div>

      <div className="results-section">
        {loading && <p className="loading">Thinking... 🧠</p>}
        {error && <p className="error">{error}</p>}
        
        {recommendations.length > 0 && (
          <div className="movie-grid">
            {recommendations.map((movie, index) => (
              <div key={index} className="movie-card">
                <span className="rank">#{index + 1}</span>
                <h3>{movie}</h3>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App