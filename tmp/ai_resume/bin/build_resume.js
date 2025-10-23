// node imports
import Fs from 'node:fs';

// npm imports
import Puppeteer from 'puppeteer';
import { marked as Marked } from 'marked';
import FrontMatter from 'front-matter';

const __dirname = new URL('.', import.meta.url).pathname;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//	
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class PdfUtils {
	/**
	 * Use puppeteer to print the resume as a PDF
	 * 
	 * @param {string} pageUrl - The URL of the page to print
	 * @returns {Promise<Uint8Array<ArrayBufferLike>>}
	 */
	static async printPDF(pageUrl) {
		const browser = await Puppeteer.launch();
		const page = await browser.newPage();
		await page.goto(pageUrl, { waitUntil: 'networkidle0' });
		const pdf = await page.pdf({ format: 'A4' });
		await browser.close();
		return pdf;
	}
}

class TokenUtils {

	/**
	 * @param {import('marked').TokensList} mdTokens
	 * @param {string} title_text
	 * @param {number} title_depth
	 */
	static find_section(mdTokens, title_text, title_depth) {
		const section_index = mdTokens.findIndex(token => token.type === 'heading' && token.depth === title_depth && token.text.toLowerCase() === title_text.toLowerCase());
		if (section_index === -1) {
			return null;
		}
		// Find the next heading of the same or higher depth
		let end_index = mdTokens.length;
		for (let i = section_index + 1; i < mdTokens.length; i++) {
			const token = mdTokens[i];
			if (token.type === 'heading' && token.depth <= title_depth) {
				end_index = i;
				break;
			}
		}
		return mdTokens.slice(section_index + 1, end_index);
	}

	/**
	 * 
	 * @param {import('marked').TokensList} mdTokens 
	 * @param {string} title_text 
	 * @param {number} title_depth 
	 * @returns {string} HTML string of the section
	 */
	static section_to_html(mdTokens, title_text, title_depth) {
		const sectionTokens = TokenUtils.find_section(mdTokens, title_text, title_depth);
		console.assert(sectionTokens !== null, `${title_text} section not found`);
		if (sectionTokens === null) {
			throw new Error(`${title_text} section not found`);
		}
		console.log(JSON.stringify(sectionTokens, null, 2));
		const sectionHtml = Marked.parser(sectionTokens);
		return sectionHtml;
	}

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//	Main 
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

async function main() {
	/**
	 * @type {Object<string, any>}
	 */
	let ejsData = {}


	// read the markdown file
	const md_path = `${__dirname}/../resume_webdev.md`
	const fileContent = await Fs.promises.readFile(md_path, 'utf-8');

	// parse the front matter
	const frontMatterResult = FrontMatter(fileContent);
	frontMatterResult.attributes; // Object containing the front matter data

	// parse the markdown content
	const md_content = frontMatterResult.body;
	const md_tokens = Marked.lexer(md_content);


	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//	Push front-matter in ejsData
	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	ejsData.fm_attributes = frontMatterResult.attributes;

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//	Build ejs_data from md_tokens
	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	ejsData.skillsSectionHtml = TokenUtils.section_to_html(md_tokens, 'Skills', 2);



	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//	
	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// read the ejs template
	const ejs_template_path = `${__dirname}/../templates/resume_template.html.ejs`;
	const ejs_template = await Fs.promises.readFile(ejs_template_path, 'utf-8');

	// render the ejs template with the data
	const ejs = await import('ejs');
	const rendered_html = ejs.render(ejs_template, ejsData);

	// write the rendered html to a file
	const html_basename = 'resume_webdev.html';
	const output_html_path = `${__dirname}/../output/${html_basename}`;
	await Fs.promises.writeFile(output_html_path, rendered_html);

	console.log('Resume HTML generated at:', output_html_path);

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//	
	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	const serverUrl = 'http://localhost:8080';
	const html_url = `${serverUrl}/output/${html_basename}`;
	const pdf_buffer = await PdfUtils.printPDF(html_url)

	const pdf_path = `${__dirname}/../output/resume_webdev.pdf`;
	await Fs.promises.writeFile(pdf_path, pdf_buffer);

	console.log('Resume PDF generated at:', pdf_path);

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//	
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void main();