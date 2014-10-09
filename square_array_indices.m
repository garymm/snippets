function indices = square_array_indices(n)
	left = repmat(transpose(1:n), n, 1);
	right = reshape(repmat(1:n, n, 1), [n*n 1]);
	indices = [left right];
end