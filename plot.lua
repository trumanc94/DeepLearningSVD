require 'torch'
-- Define tensor data parameters
numMovies = 3952
numUsers = 6040
numRatings = 1000209 -- Number of actual ratings
numFeats = 40

-- Slightly Modified version of: 
--   https://github.com/clementfarabet/manifold/blob/master/init.lua
-- 		(requires qlua)
function draw_text_map(X, words, inp_map_size, inp_font_size, reduction)
  
  -- input options:
  local map_size  = inp_map_size or 512
  local font_size = inp_font_size or 9
  
  -- check inputs are correct:
  local N = X:size(1)
  if X:nDimension() ~= 2 or X:size(2) ~= 2 then
    error('This function is designed to operate on 2D embeddings only.')
  end
  if X:size(1) ~= #words then
    error('Number of words should match the number of rows in X.')
  end
  
  -- prepare image for rendering:
  require 'image'
  require 'qtwidget'
  require 'qttorch'
  local win = qtwidget.newimage(map_size, map_size)
  
  --render the words:
  for key,val in pairs(words) do
    win:setfont(qt.QFont{serif = false, size = fontsize})
    local x = math.floor(X[key][1] * map_size/reduction + map_size/2 - 50)
    local y = math.floor(X[key][2] * map_size/reduction + map_size/2)
    win:moveto(x,y)
    win:show(val)
  end
  
  -- render to tensor:
  local map_im = win:image():toTensor()
  
  -- return text map:
  return map_im
end
--print( userFeatures )
--print( movieFeatures )


-- Display movie scatter plot
m = require 'manifold'
IdArr = torch.Tensor(numMovies)
TitleArr = {}

-- If binary data file does not exist, load rating data from MovieLens data file
if not paths.filep("movieLabels.t7") then
	-- File format: userId::movieId::rating::date
	file = io.open("./ml-1m/movies.dat", "r")
	
	i = 1
	for line in file:lines() do
		movieId, movieLabel = line:match("([^::]+)::([^::]+)")
		id = tonumber(movieId)
		if i ~= id then
			while i < id do
				IdArr[i] = i
				TitleArr[i] = "Not Listed"
				print( "Title ", i, " is unlisted" )
				i = i + 1
			end
		end
		IdArr[i] = i
		TitleArr[id] = movieLabel
		i = i + 1
		collectgarbage()
	end
	movieLabels = 
	{
		ids = IdArr,
		titles = TitleArr
	}
	file:close()	
	torch.save("movieLabels.t7", movieLabels)
-- Otherwise, load data from binary file
else
	movieLabels = torch.load("movieLabels.t7")	
end

if not paths.filep("movieFeatures.t7") then
	print( "movieFeatures.t7", " not found" )
else
	movieFeatures = torch.load( "movieFeatures.t7" )
	temp = movieFeatures:type('torch.DoubleTensor')
	if paths.filep("moviePlot.t7") then
		moviePlot = torch.load( "moviePlot.t7" )
	else
		moviePlot = m.embedding.tsne(temp, {dim=2, perplexity=30})
		torch.save( "moviePlot.t7", moviePlot )
	end
	map = draw_text_map( moviePlot, movieLabels.titles, 2048, 6, 175 )
	local gfx = require 'gfx.js'
	gfx.image(map)
end
