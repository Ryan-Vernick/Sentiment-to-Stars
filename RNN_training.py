


for tokenized_review, rating in training_reviews:
    optimizer.zero_grad()
    predicted_rating = our_RNN(tokenized_review)
    loss = loss_function (predicted_rating, known_rating)
    loss.backward()
    optimizer.step()
