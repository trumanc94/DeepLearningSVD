-- Define parameters
numMovies = 3952
numUsers = 6040
numRatings = 1000209 -- Number of actual ratings
numFeats = 40
initialFeatureVal = 0.1
learningRate = 0.01
lambda = 0.015

-- Create tensors for ratings, users, and movies
ratings = torch.FloatTensor(numRatings)
user = torch.IntTensor(numRatings)
movie = torch.IntTensor(numRatings)

-- Load data
-- If binary data file does not exist, load rating data from MovieLens data file
if not paths.filep("saveDataTable.t7") then
	-- File format: userId::movieId::rating::date
	file = io.open("./ml-1m/ratings.dat", "r")
	i = 1
	for line in file:lines() do
		userId, movieId, rating = line:match("([^::]+)::([^::]+)::([^::]+)")
		ratings[i] = rating
		user[i] = userId
		movie[i] = movieId
		i = i + 1
	end
	file:close()	

	-- Store data into table
	saveDataTable =
	{
		savedRatings = ratings,
		savedUsers = user,
		savedMovies = movie		
	}	
	torch.save("saveDataTable.t7", saveDataTable)

-- Otherwise, load data from binary file
else
	loaded = torch.load("saveDataTable.t7")	
	saveDataTable =
	{
		savedRatings = loaded.savedRatings,
		savedUsers = loaded.savedUsers,
		savedMovies = loaded.savedMovies
	}	
end

--[[ Calculate average movie ratings
averageRatings = torch.FloatTensor(numMovies, 2):fill(0.0)
for i=1,numRatings do
	index = saveDataTable.savedMovies[i]
	averageRatings[index][1] = averageRatings[index][1] + saveDataTable.savedRatings[i]
	averageRatings[index][2] = averageRatings[index][2] + 1
end

for i=1, numMovies do
	averageRatings[i][1] = averageRatings[i][1] / averateRatings[i][2]
end
--]]

-- Create users and movies matrices for SVD: X = USV'
-- U = user matrix, V' = movie matrix, S = orthogonal matrix
-- Rating = U x V'
userFeatures = torch.FloatTensor(numUsers, numFeats):fill(initialFeatureVal)
movieFeatures = torch.FloatTensor(numMovies, numFeats):fill(initialFeatureVal)


-- Train functions: train_by_features(), train_all_features()
-- Train each feature individually
function train_by_features()
	-- Define function parameters
	MIN_IMPROVEMENT = 0.0001
	MIN_EPOCHS = 10
	rmse = 0
	last_rmse = 0

	-- Train each feature
	for f = 1,numFeats do
		-- Train for specified number of epochs, or continue until no improvement
		epoch = 0
		while (epoch < MIN_EPOCHS) or (rmse < last_rmse - MIN_IMPROVEMENT) do
			-- Store last RMSE
			last_rmse = rmse

			-- Calculate squared error
			squared_error = 0
			for i=1, numRatings do
				-- Display progress
				xlua.progress(i, numRatings)

				local movieID = saveDataTable.savedMovies[i]
				local userID = saveDataTable.savedUsers[i]

				-- Cache off old feature values
				local cf = userFeatures[userID][f]
				local mf = movieFeatures[movieID][f]
				
				-- Calculate predicted rating (output must be between 1 and 5)
				local output = mf * cf
				if output > 5 then
					output = 5.0
				elseif output < 1 then
					output = 1.0
				end

				-- Calculate error: rating - predicted rating
			  	local err = saveDataTable.savedRatings[i] - output
				  
				-- Calculate squared error
				squared_error = squared_error + err^2

				-- Cross-train the features
				userFeatures[userID][f] = cf + learningRate * (err * mf - lambda * cf)
				movieFeatures[movieID][f] = mf + learningRate * (err * cf - lambda * mf)
			end

			-- Calculate root mean squared error
			rmse = (squared_error / numRatings)^0.5

			epoch = epoch + 1
		end

		-- Print RMSE for finished feature
		print('RMSE =', rmse)
		print('Feature=', f)
	end
end

-- Train all features together
function train_all_features()
	-- Train each feature
	for f = 1,numFeats do
		-- Display progress
		xlua.progress(f, numFeats)

		-- Calculate squared error
		squared_error = 0
		for i=1, numRatings do
			local movieID = saveDataTable.savedMovies[i]
			local userID = saveDataTable.savedUsers[i]

			-- Cache off old feature values
			local cf = userFeatures[userID][f]
			local mf = movieFeatures[movieID][f]
				
			-- Calculate predicted rating
			local output = mf * cf
			if output > 5 then
				output = 5.0
			elseif output < 1 then
				output = 1.0
			end

			-- Calculate error: rating - predicted rating
		  	local err = saveDataTable.savedRatings[i] - output
				  
			-- Calculate squared error
			squared_error = squared_error + err^2

			-- Cross-train the features
			userFeatures[userID][f] = cf + learningRate * (err * mf - lambda * cf)
			movieFeatures[movieID][f] = mf + learningRate * (err * cf - lambda * mf)
		end

		-- Calculate root mean squared error
		rmse = (squared_error / numRatings)^0.5
	end

	-- Print RMSE 
	print('RMSE =', rmse)
end

-- Train by features
--train_by_features()


-- Train all features
for epoch=1,40 do
	train_all_features()
end

-- Save trained feature matrices
torch.save("userFeatures.t7", userFeatures)
torch.save("movieFeatures.t7", movieFeatures)
