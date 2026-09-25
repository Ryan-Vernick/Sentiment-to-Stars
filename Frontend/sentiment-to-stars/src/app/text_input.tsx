'use client'

import React from "react"

import styles from "./text_input.module.css";
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
    const calculateRating = () => {
        if (reviewText.length > 0) {
            //use our ML algorithm on the review text
            setComputedRating(1) // 1 is a placeholder for the result
        } else {
            setComputedRating(null)
        }
    }
    return (
        <button type="button" className={styles.submitButton} onClick={calculateRating}>Generate Rating</button>
    )
}


export function OneStar({fillLevel, computedRating}: {fillLevel: number, computedRating: number | null}) {
    const roundedLevel = Math.round(fillLevel * 2) / 2
    const clipPercentage = (1 - roundedLevel) * 100 + '%'
    return (
        <div className={styles.starHolder}>
            <FontAwesomeIcon className={styles.starFill} icon={fasStar} style={{color: computedRating ? '#fde1a0' : '#f2f2f2', clipPath: fillLevel < 1 ? `inset(0 ${clipPercentage} 0 0)` : ''}} />
            <FontAwesomeIcon className={styles.starOutline} icon={farStar} style={{color: computedRating ? '#e7b851' : '#929292'}} />
            {/* <img className={styles.starFill} src='/star-fill.svg' style={{clipPath: fillLevel < 1 ? `inset(0 ${clipPercentage} 0 0)` : ''}}/>
            <img className={styles.starOutline} src='/star-outline.svg'/> */}
        </div>
    )
}

export function DisplayStars({appState}: {appState: appStateType}) {
    const numStars = appState.numStars
    const scale = appState.scale
    const computedRating = appState.computedRating
    const numericalRating = computedRating ? computedRating * scale : 0
    const numFilled = computedRating ? numericalRating * numStars / scale : numStars
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