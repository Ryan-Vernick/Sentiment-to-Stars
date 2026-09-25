'use client'

import styles from "./page.module.css";
import TextInput, { DisplayStars, NavBar, SubmitReview } from "./text_input";

import '@fontsource-variable/league-spartan';
import { useState } from "react";

export type appStateType = {
  numStars: number;
  scale: number;
  reviewInput: string;
  setReviewInput: (v: string) => void;
  computedRating: number | null;
  setComputedRating: (v: number | null) => void;
}

export default function Home() {
  const [numStars, setNumStars] = useState<number>(5);
  const [scale, setScale] = useState<number>(10);
  const [reviewInput, setReviewInput] = useState<string>('');
  const [computedRating, setComputedRating] = useState<number | null>(null);

  const appState: appStateType = {
    numStars,
    scale,
    reviewInput,
    setReviewInput,
    computedRating,
    setComputedRating
  }

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <NavBar/>
        <div className={styles.content}>
          <DisplayStars appState={appState}/>
          <TextInput appState={appState}/>
          <SubmitReview appState={appState}/>
        </div>
      </main>
    </div>
  );
}
