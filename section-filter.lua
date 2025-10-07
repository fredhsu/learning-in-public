-- section-filter.lua
-- Pandoc filter to wrap h2 elements and their content in <section> tags
-- and wrap the entire document in <article> tags

local sections = {}
local current_section = nil

function Pandoc(doc)
	local new_blocks = {}

	-- Add opening article tag at the beginning
	table.insert(new_blocks, pandoc.RawBlock("html", "<article>"))

	for i, block in ipairs(doc.blocks) do
		if block.t == "Header" and block.level == 2 then
			-- Close previous section if it exists
			if current_section then
				table.insert(new_blocks, pandoc.RawBlock("html", "</section>"))
			end

			-- Start new section
			table.insert(new_blocks, pandoc.RawBlock("html", "<section>"))
			table.insert(new_blocks, block)
			current_section = true
		else
			table.insert(new_blocks, block)
		end
	end

	-- Close the last section if it exists
	if current_section then
		table.insert(new_blocks, pandoc.RawBlock("html", "</section>"))
	end

	-- Add closing article tag at the end
	table.insert(new_blocks, pandoc.RawBlock("html", "</article>"))

	return pandoc.Pandoc(new_blocks, doc.meta)
end

