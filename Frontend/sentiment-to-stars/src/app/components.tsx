'use client'

import React from "react"
import { useState } from "react";

import styles from "./components.module.css";
import { appStateType } from "./page";

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faStar as fasStar } from '@fortawesome/free-solid-svg-icons'
import { faStar as farStar } from '@fortawesome/free-regular-svg-icons'

export default function TextInput({appState}: {appState: appStateType}) {
    const reviewText = appState.reviewInput
    const setReviewText = appState.setReviewInput

    const updateReviewText = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        if (event.target) {
            setReviewText(event.target.value)
        }
    }
    return (
        <div className={styles.inputSection}>
            <label htmlFor='reviewInput' className={styles.inputLabel}>Enter review text:</label>
            <textarea id='reviewInput' value={reviewText} className={styles.textInput} onChange={updateReviewText} spellCheck/>
        </div>
    )
}

export function SubmitReview({appState}: {appState: appStateType}) {
    const reviewText = appState.reviewInput
    const setComputedRating = appState.setComputedRating
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Function called on button click
    const handlePredict = async () => {
        if (!reviewText.trim()) {
            setComputedRating(null)
            setError('Enter review text before generating a rating.')
            return
        }

        setIsLoading(true)
        setError(null)
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ input_value: reviewText }),
            })

            const data = await response.json()
            if (!response.ok) {
                throw new Error(data.error ?? 'The rating request failed.')
            }
            if (typeof data.result !== 'number' || !Number.isFinite(data.result)) {
                throw new Error('The model returned an invalid rating.')
            }
            setComputedRating(data.result)
        } catch (error) {
            console.error('Error connecting to ML backend:', error)
            setComputedRating(null)
            setError(error instanceof Error ? error.message : 'Could not connect to the model.')
        } finally {
            setIsLoading(false)
        }
    }
    return (
        <>
            <button type="button" className={styles.submitButton} onClick={handlePredict} disabled={isLoading}>
                {isLoading ? 'Generating...' : 'Generate Rating'}
            </button>
            {error && <p role="alert">{error}</p>}
        </>
    )
}


export function OneStar({fillLevel, computedRating}: {fillLevel: number, computedRating: number | null}) {
    const roundedLevel = Math.round(fillLevel * 2) / 2
    const clipPercentage = (1 - roundedLevel) * 100 + '%'
    return (
        <div className={styles.starHolder}>
            <FontAwesomeIcon className={styles.starFill} icon={fasStar} style={{color: computedRating ? '#fde1a0' : '#f2f2f2', clipPath: fillLevel < 1 ? `inset(0 ${clipPercentage} 0 0)` : ''}} />
            <FontAwesomeIcon className={styles.starOutline} icon={farStar} style={{color: computedRating ? '#e7b851' : '#aeaeae'}} />
            {/* <img className={styles.starFill} src='/star-fill.svg' style={{clipPath: fillLevel < 1 ? `inset(0 ${clipPercentage} 0 0)` : ''}}/>
            <img className={styles.starOutline} src='/star-outline.svg'/> */}
        </div>
    )
}

export function DisplayStars({appState}: {appState: appStateType}) {
    const numStars = appState.numStars
    const scale = appState.scale
    const computedRating = appState.computedRating
    const numericalRating = computedRating ? Math.round((computedRating * scale) * 2) / 2 : 0
    const numFilled = computedRating ? computedRating * numStars : numStars
    let fillLeft = numFilled

    const stars = []

    for (let i = 0; i < numStars; i++) {
        const fillLevel = fillLeft >= 1 ? 1 : fillLeft
        stars.push(<OneStar key={i} fillLevel={fillLevel} computedRating={computedRating}/>)
        
        fillLeft--;
        if (fillLeft < 0) fillLeft = 0
    }

    return (
        <div className={styles.results}>
            {/* <h3 className={styles.numericalRating}>{numFilled}/{numStars}</h3> */}
            <h3 className={styles.numericalRating}>{computedRating ? numericalRating : '?'}/{scale}</h3>
            <div className={styles.stars}>
                {stars}
            </div>
        </div>
    )
}

export function NavBar() {
    return (
        <div className={styles.navBar}>
            <h1 className={styles.title}>Sentiment to St<FontAwesomeIcon className={styles.titleStar} icon={farStar}/>rs</h1>
        </div>
    )
}